class InsufficientDataError(Exception):
    """
    Custom exception for insufficient data length.
    """
    def __init__(self, seq_len, pred_len, actual_len):
        message = (
            f"Not enough data for sequence and prediction length. "
            f"The input length should be at least: {seq_len + pred_len + 1}, "
            f"but it is: {actual_len}."
        )
        super().__init__(message)
