"""AU adapter components for expression-conditioned image generation.

Contains AUEncoder, IdentityAdapter, AUIPAttnProcessor, and helpers for
setting up UNet processors, loading IP-Adapter weights, and save/load utilities.
"""

import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from safetensors.torch import load_file, save_file

from .constants import BP4D_AU_COLUMNS

NUM_AUS: int = len(BP4D_AU_COLUMNS)  # 23


class AUEncoder(nn.Module):
    """Encode AU intensity values to cross-attention conditioning tokens.

    The raw AU values are concatenated with a learned MLP projection before
    the final linear maps them to tokens — a skip connection that ensures the
    raw signal is always visible regardless of what the MLP learns.

    Maps (B, NUM_AUS) -> (B, num_tokens, dim).

    Also provides `encode` for classifier-free guidance: batches
    unconditional (zeros) and conditional embeddings together along dim=0,
    matching SD's CFG pattern.
    """

    def __init__(
        self,
        num_aus: int = NUM_AUS,
        hidden: int = 256,
        dim: int = 768,
        num_tokens: int = 4,
    ) -> None:
        """Initialise AUEncoder.

        Args:
            num_aus:
              Number of action units (input dimension).
            hidden:
              Hidden dimension of the MLP.
            dim:
              Cross-attention dimension (output per token).
            num_tokens:
              Number of conditioning tokens to produce.
        """
        super().__init__()
        self.num_aus = num_aus
        self.num_tokens = num_tokens
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(in_features=num_aus, out_features=hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # projects [raw_aus | mlp_out] -> tokens
        self.to_tokens = nn.Linear(
            in_features=num_aus + hidden, out_features=dim * num_tokens
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, au_values: torch.Tensor) -> torch.Tensor:
        """Encode AU values to conditioning tokens.

        Args:
            au_values:
              AU intensity tensor of shape (B, num_aus).

        Returns:
            Conditioning tokens of shape (B, num_tokens, dim).
        """
        projected = self.mlp(au_values)
        x = torch.cat([au_values, projected], dim=-1)  # skip connection
        x = self.to_tokens(x)
        x = x.reshape(-1, self.num_tokens, self.dim)
        return self.norm(x)

    def encode(self, au_values: torch.Tensor) -> torch.Tensor:
        """Encode AU values for classifier-free guidance.

        Batches unconditional (zero AU) and conditional embeddings together
        along dim=0, matching SD's CFG pattern. `au_scale` in
        `AUIPAttnProcessor` controls expression strength at inference.

        Args:
            au_values:
              AU intensity tensor of shape (B, num_aus).

        Returns:
            Batched tensor of shape (2*B, num_tokens, dim) — uncond first,
            cond second.
        """
        cond = self.forward(au_values)
        uncond = self.forward(torch.zeros_like(au_values))
        return torch.cat([uncond, cond], dim=0)


class IdentityAdapter(nn.Module):
    """Personalise AU tokens using a subject's ArcFace embedding.

    Projects the ArcFace embedding to per-token scale and shift vectors and
    applies them: `output = au_tokens * (1 + scale) + shift`.

    Zero-initialised so it acts as identity at the start of training,
    letting the AU encoder converge before subject-specific corrections
    are learned.
    """

    def __init__(self, dim: int = 768, faceid_dim: int = 512) -> None:
        """Initialise IdentityAdapter.

        Args:
            dim:
              Token dimension (must match AUEncoder dim).
            faceid_dim:
              Dimension of the ArcFace embedding.
        """
        super().__init__()
        self.to_scale = nn.Linear(in_features=faceid_dim, out_features=dim)
        self.to_shift = nn.Linear(in_features=faceid_dim, out_features=dim)
        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(
        self, au_tokens: torch.Tensor, faceid_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Apply identity-conditioned scale and shift to AU tokens.

        Args:
            au_tokens:
              AU conditioning tokens of shape (B, num_tokens, dim).
            faceid_embeds:
              Normalised ArcFace embeddings of shape (B, faceid_dim).

        Returns:
            Personalised tokens of shape (B, num_tokens, dim).
        """
        scale = self.to_scale(faceid_embeds).unsqueeze(1)  # (B, 1, dim)
        shift = self.to_shift(faceid_embeds).unsqueeze(1)  # (B, 1, dim)
        return au_tokens * (1 + scale) + shift


class MLPProjModel(nn.Module):
    """Project ArcFace embedding to IP-Adapter face tokens.

    Loaded from the pretrained IP-Adapter-FaceID checkpoint (`image_proj` key).
    Maps (B, id_embeddings_dim) -> (B, num_tokens, cross_attention_dim).
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        id_embeddings_dim: int = 512,
        num_tokens: int = 16,
    ) -> None:
        """Initialise MLPProjModel with given dimensions."""
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim * 2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim * 2, cross_attention_dim * num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds: torch.Tensor) -> torch.Tensor:
        """Project ArcFace embeddings to face tokens.

        Args:
            id_embeds: ArcFace embeddings of shape (B, id_embeddings_dim).

        Returns:
            Face tokens of shape (B, num_tokens, cross_attention_dim).
        """
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class SelfAttnProcessor(nn.Module):
    """nn.Module wrapper around AttnProcessor2_0 for self-attention blocks.

    Being an nn.Module lets all processors sit in nn.ModuleList for bulk
    IP-Adapter weight loading (AttnProcessor2_0 is a plain Python class).
    Conditioning kwargs are named explicitly so diffusers does not emit
    "are not expected and will be ignored" warnings.
    """

    def __init__(self) -> None:
        """Initialise SelfAttnProcessor wrapping AttnProcessor2_0."""
        super().__init__()
        self._proc = AttnProcessor2_0()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        id_embedding: torch.Tensor | None = None,
        id_scale: float = 1.0,
        au_embedding: torch.Tensor | None = None,
        au_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass — delegates to AttnProcessor2_0, ignoring conditioning kwargs.

        Returns:
            Attention output tensor.
        """
        return self._proc(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )


class AUIPAttnProcessor(nn.Module):
    """Full cross-attention processor handling text, IP-Adapter, and AU conditioning.

    Implements the complete attention forward pass using
    `F.scaled_dot_product_attention`, combining text, IP-Adapter (identity),
    and AU (expression) conditioning in one processor with a single `to_out`
    call. IP-Adapter projections are loaded from pretrained weights;
    AU projections are zero-initialised and trained from scratch.

    Conditioning is passed via `cross_attention_kwargs` at UNet call time:
        `cross_attention_kwargs={
            "id_embedding": ...,  # (B, num_id_tokens, dim)
            "id_scale": 0.6,
            "au_embedding": ...,  # (B, num_au_tokens, dim) or (2B, ...) for CFG
            "au_scale": 1.0,
        }`
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int = 768) -> None:
        """Initialise AUIPAttnProcessor.

        Args:
            hidden_size:
              Output dimension of the attention block (from UNet block config).
            cross_attention_dim:
              Dimension of the cross-attention conditioning (IP and AU tokens).
        """
        super().__init__()
        # Identity (IP-Adapter) projections — weights loaded from pretrained checkpoint
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        # AU projections — zero-init so they start as a no-op
        self.to_k_au = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_au = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        nn.init.zeros_(self.to_k_au.weight)
        nn.init.zeros_(self.to_v_au.weight)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        id_embedding: torch.Tensor | None = None,
        id_scale: float = 1.0,
        au_embedding: torch.Tensor | None = None,
        au_scale: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """Full attention forward: text + IP-Adapter + AU, single to_out.

        Args:
            attn: Parent Attention module.
            hidden_states: Query hidden states (B, S, C).
            encoder_hidden_states: Text key/value states or None for self-attn.
            attention_mask: Optional attention mask.
            temb: Optional timestep embedding for spatial norm.
            id_embedding: Identity face tokens (B, num_id_tokens, dim).
            id_scale: Identity conditioning strength.
            au_embedding: AU tokens (B, num_au_tokens, dim) or (2B, ...) for CFG.
            au_scale: Expression conditioning strength.
            kwargs: Additional keyword arguments.

        Returns:
            Output tensor (B, S, C).
        """
        # ======== Start diffusers (AttnProcessor2_0) ========
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        c = h = w = 0
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        b, s, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, s, b)
            attention_mask = attention_mask.view(
                b, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        query = query.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(b, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(b, -1, attn.heads, head_dim).transpose(1, 2)

        # text cross-attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            b, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        # ======== End diffusers (AttnProcessor2_0) ========

        # ======== IP-Adapter: identity conditioning (to_k_ip / to_v_ip) ========
        if id_embedding is not None:
            id_embedding = id_embedding.to(dtype=query.dtype)
            ip_key = (
                self.to_k_ip(id_embedding)
                .view(b, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            ip_value = (
                self.to_v_ip(id_embedding)
                .view(b, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            ip_hidden = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            ip_hidden = (
                ip_hidden.transpose(1, 2)
                .reshape(b, -1, attn.heads * head_dim)
                .to(query.dtype)
            )
            hidden_states = hidden_states + id_scale * ip_hidden
        # ======== End IP-Adapter ========

        # ======== AU adapter: expression conditioning (to_k_au / to_v_au) ========
        if au_embedding is not None:
            au_embedding = au_embedding.to(dtype=query.dtype)
            # CFG-batched: (2B, num_tokens, dim) → split, interpolate, recombine
            if au_embedding.shape[0] == 2 * b:
                uncond, cond = au_embedding.chunk(2, dim=0)
                au_embedding = uncond + au_scale * (cond - uncond)

            au_key = (
                self.to_k_au(au_embedding)
                .view(b, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            au_value = (
                self.to_v_au(au_embedding)
                .view(b, -1, attn.heads, head_dim)
                .transpose(1, 2)
            )
            au_hidden = F.scaled_dot_product_attention(
                query, au_key, au_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            au_hidden = (
                au_hidden.transpose(1, 2)
                .reshape(b, -1, attn.heads * head_dim)
                .to(query.dtype)
            )
            hidden_states = hidden_states + au_scale * au_hidden
        # ======== end AU adapter ========

        # ======== Start diffusers (AttnProcessor2_0) ========
        # linear proj + dropout — called once on the combined output
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # ======== End diffusers (AttnProcessor2_0) ========
        return hidden_states


def setup_unet_processors(unet: UNet2DConditionModel) -> nn.ModuleDict:
    """Replace all cross-attention processors with AUIPAttnProcessor.

    Self-attention (attn1) blocks get the standard AttnProcessor2_0.
    Uses UNet block config to get the correct hidden_size per block.

    Args:
        unet: The UNet to configure.

    Returns:
        ModuleDict of AUIPAttnProcessors keyed by sanitised block name.
    """
    procs: dict[str, t.Any] = {}
    au_procs: dict[str, AUIPAttnProcessor] = {}

    for name in unet.attn_processors:
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[-1]

        if cross_attention_dim is not None:
            proc = AUIPAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )
            au_procs[name] = proc
            procs[name] = proc
        else:
            procs[name] = SelfAttnProcessor()

    unet.set_attn_processor(procs)
    return nn.ModuleDict({k.replace(".", "_"): v for k, v in au_procs.items()})


def load_ip_adapter_weights(
    unet: UNet2DConditionModel, path: str, num_tokens: int = 16
) -> "MLPProjModel":
    """Load pretrained IP-Adapter-FaceID weights and return the face projector.

    Loads `to_k_ip`/`to_v_ip` into the UNet's `AUIPAttnProcessor`s via
    `ModuleList + strict=False` (self-attention `SelfAttnProcessor`s are
    skipped automatically). Builds and returns an `MLPProjModel` loaded with
    the checkpoint's `image_proj` weights.

    Args:
        unet: UNet with processors set up via `setup_unet_processors`.
        path: Path to the IP-Adapter-FaceID `.bin` checkpoint.
        num_tokens: Number of face tokens (16 for portrait-v11).

    Returns:
        `MLPProjModel` with pretrained weights, ready for inference.
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    # Load to_k_ip / to_v_ip — strict=False skips SelfAttnProcessors that
    # have no IP-Adapter keys in the checkpoint.
    ip_layers = nn.ModuleList(unet.attn_processors.values())
    ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
    # Build the face projector and load its pretrained weights
    face_proj = MLPProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        num_tokens=num_tokens,
    )
    face_proj.load_state_dict(state_dict["image_proj"])
    return face_proj


def prefixed(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """Prefix all keys in a state dict.

    Args:
        state_dict: Model state dict.
        prefix: String prepended to every key.

    Returns:
        New dict with keys `f"{prefix}.{k}"`.
    """
    return {f"{prefix}.{k}": v for k, v in state_dict.items()}


def save_au_adapter(
    au_encoder: AUEncoder,
    identity_adapter: IdentityAdapter,
    au_procs: nn.ModuleDict,
    path: str,
) -> None:
    """Serialise all trainable AU adapter weights to a safetensors file.

    IP-Adapter weights (`to_k_ip`, `to_v_ip`) are excluded — they are
    pretrained and loaded separately via `load_ip_adapter_weights`.

    Args:
        au_encoder: Trained AUEncoder.
        identity_adapter: Trained IdentityAdapter.
        au_procs: Trained AUIPAttnProcessors (as ModuleDict).
        path: Destination path for the safetensors file.
    """
    procs_sd = {
        k: v
        for k, v in au_procs.state_dict().items()
        if not k.endswith(("to_k_ip.weight", "to_v_ip.weight"))
    }
    flat: dict[str, torch.Tensor] = {}
    flat.update(prefixed(au_encoder.state_dict(), "au_encoder"))
    flat.update(prefixed(identity_adapter.state_dict(), "identity_adapter"))
    flat.update(prefixed(procs_sd, "au_procs"))
    save_file(tensors=flat, filename=path)


def load_au_adapter(
    path: str, unet: UNet2DConditionModel, ip_adapter_path: str
) -> tuple[AUEncoder, IdentityAdapter, nn.ModuleDict]:
    """Load AU adapter weights from a safetensors file.

    Sets up fresh AUIPAttnProcessors, loads IP-Adapter weights, then loads
    AU adapter weights on top.

    Args:
        path: Path to the safetensors file produced by save_au_adapter.
        unet: The UNet to inject processors into.
        ip_adapter_path: Path to the IP-Adapter checkpoint.

    Returns:
        Tuple of (au_encoder, identity_adapter, au_procs).
    """
    flat = load_file(filename=path)

    def _strip(d: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        n = len(prefix) + 1
        return {k[n:]: v for k, v in d.items() if k.startswith(prefix + ".")}

    au_encoder = AUEncoder()
    au_encoder.load_state_dict(_strip(flat, "au_encoder"))

    identity_adapter = IdentityAdapter()
    identity_adapter.load_state_dict(_strip(flat, "identity_adapter"))

    au_procs = setup_unet_processors(unet)
    load_ip_adapter_weights(unet, ip_adapter_path)
    au_procs.load_state_dict(_strip(flat, "au_procs"), strict=False)

    return au_encoder, identity_adapter, au_procs
