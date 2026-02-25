"""Identity Loss."""

import collections.abc as c
import typing as t

from deepface import DeepFace
from numpy.typing import NDArray

from ..enums import IdentityBackend, IdentityModel

ImageType = str | NDArray[t.Any] | t.IO[bytes]
BatchImageType = list[str] | list[NDArray[t.Any]] | list[t.IO[bytes]]
EmbeddingType = c.Sequence[dict[str, t.Any]] | c.Sequence[list[dict[str, t.Any]]]


class IdentityLoss:
    """Identity loss."""

    def __init__(
        self,
        model_name: IdentityModel = IdentityModel.VGG_FACE,
        detector_backend: IdentityBackend = IdentityBackend.OPENCV,
    ) -> None:
        """Initialize the identity loss.

        Args:
            model_name:
                The name of the model to use.
            detector_backend:
                The backend to use for face detection.
        """
        self.model_name: str = IdentityModel(model_name).value
        self.detector_backend: str = IdentityBackend(detector_backend).value

    def verify(
        self, first_image: ImageType, second_image: ImageType, **kwargs
    ) -> dict[str, t.Any]:
        """Verify that the two images are of the same person.

        Args:
            first_image:
                The first image to compare.
            second_image:
                The second image to compare.
            **kwargs:
                Additional keyword arguments to pass to the verification model.

        Returns:
            A dictionary containing the verification result.
        """
        return DeepFace.verify(
            img1_path=first_image,
            img2_path=second_image,
            model_name=self.model_name,
            **kwargs,
        )

    def embedding(self, image: ImageType | BatchImageType, **kwargs) -> EmbeddingType:
        """Get the embedding of the faces in the image.

        Args:
            image:
                The image or list of images to get the embedding(s) of.
            **kwargs:
                Additional keyword arguments to pass to the embedding model.

        Returns:
            The embedding of the faces in the image.
        """
        return DeepFace.represent(img_path=image, model_name=self.model_name, **kwargs)

    def get_faces(self, image: ImageType | BatchImageType, **kwargs) -> EmbeddingType:
        """Get the faces in the image.

        Args:
            image:
                The image or list of images to get the faces from.
            **kwargs:
                Additional keyword arguments to pass to the face detection model.

        Returns:
            The faces in the image.
        """
        return DeepFace.extract_faces(
            img_path=image, detector_backend=self.detector_backend, **kwargs
        )

    def loss(self, first_image: ImageType, second_image: ImageType, **kwargs) -> None:
        """Get the loss between the two images.

        Args:
            first_image:
                The first image to compare.
            second_image:
                The second image to compare.
            **kwargs:
                Additional keyword arguments to pass to the verification model.

        Raises:
            NotImplementedError: If the input images are not of the same size.
        """
        raise NotImplementedError("Implement this method")
