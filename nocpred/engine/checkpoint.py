from detectron2.checkpoint import DetectionCheckpointer
from fvcore.common.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
    _filter_reused_missing_keys,
    _IncompatibleKeys,
)


def configure_checkpointer(checkpointer: DetectionCheckpointer):
    def _log_incompatible_keys(self: DetectionCheckpointer, incompatible: _IncompatibleKeys):
        """
        Log information about the incompatible keys returned by ``_load_model``.
        """
        for k, shape_checkpoint, shape_model in incompatible.incorrect_shapes:
            self.logger.warning(
                "Skip loading parameter '{}' to the model due to incompatible "
                "shapes: {} in the checkpoint but {} in the "
                "model! You might want to double check if this is expected.".format(
                    k, shape_checkpoint, shape_model
                )
            )

        if incompatible.missing_keys:
            missing_keys = _filter_reused_missing_keys(self.model, incompatible.missing_keys)
            missing_keys = [
                k for k in missing_keys
                if 'depth_backbone' not in k and 'normal_backbone' not in k
            ]
            if missing_keys:
                self.logger.warning(get_missing_parameters_message(missing_keys))

        if incompatible.unexpected_keys:
            self.logger.warning(get_unexpected_parameters_message(incompatible.unexpected_keys))

    checkpointer._log_incompatible_keys = _log_incompatible_keys.__get__(
        checkpointer, DetectionCheckpointer
    )
