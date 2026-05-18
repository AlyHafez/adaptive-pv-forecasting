import torch # type: ignore[import]
import torch.nn as nn # type: ignore[import]
from src.config import  residuals_config # type: ignore[import]
class ResidualCorrector(nn.Module):
    """A simple residual corrector model that takes the original TFT predictions and learns to predict the
residual errors to improve the forecasts."""

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize the ResidualCorrector model.
        Args:
            input_size (int): The number of input features (e.g., the number of quantiles or prediction dimensions).
            hidden_size (int): The number of hidden units in the fully connected layer."""
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Fully connected layer to learn residuals
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)  # Output layer to predict residuals
        self.relu2 = nn.ReLU()  # Activation function
        self.fc3 = nn.Linear(hidden_size//2, 1)  # Final output
        self.dropout = nn.Dropout(residuals_config.dropout)  # Dropout for regularization


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResidualCorrector model.
        Args:
            x (torch.Tensor): The input tensor containing the original TFT predictions.
        Returns:
            torch.Tensor: The predicted residuals to be added to the original predictions."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        residuals = self.fc3(x)
        return residuals
    