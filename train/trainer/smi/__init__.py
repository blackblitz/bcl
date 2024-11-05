"""Sequential MAP inference."""

from ...predictor import MAPPredictor
from ...training import init


class MAPMixin:
    """MAP mixin."""

    predictor_class = MAPPredictor

    def _init_params(self, key):
        """Initialize the parameters."""
        return init(key, self.model, self.mspec.in_shape)
