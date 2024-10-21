import logging
from typing import Union, Dict, Any

logger = logging.getLogger(__name__)

from phi.tools.calculator import Calculator

class EnhancedCalculator(Calculator):
    """
    Enhanced Calculator tool that extends the base Calculator from phi.tools.calculator
    """
    def _extract_result(self, result: Union[Dict[str, Any], float, int]) -> float:
        """
        Extracts the numerical result from the dictionary returned by calculator operations.
        """
        if isinstance(result, dict):
            return float(result.get('result', 0))
        return float(result)

    def add(self, a: float, b: float) -> float:
        logger.debug(f"Performing addition: {a} + {b}")
        result = super().add(a, b)
        extracted_result = self._extract_result(result)
        logger.debug(f"Addition result: {extracted_result}")
        return extracted_result

    def subtract(self, a: float, b: float) -> float:
        logger.debug(f"Performing subtraction: {a} - {b}")
        result = super().subtract(a, b)
        extracted_result = self._extract_result(result)
        logger.debug(f"Subtraction result: {extracted_result}")
        return extracted_result

    def multiply(self, a: float, b: float) -> float:
        logger.debug(f"Performing multiplication: {a} * {b}")
        result = super().multiply(a, b)
        extracted_result = self._extract_result(result)
        logger.debug(f"Multiplication result: {extracted_result}")
        return extracted_result

    def divide(self, a: float, b: float) -> float:
        logger.debug(f"Performing division: {a} / {b}")
        if b == 0:
            logger.error("Division by zero attempted")
            raise ValueError("Cannot divide by zero")
        result = super().divide(a, b)
        extracted_result = self._extract_result(result)
        logger.debug(f"Division result: {extracted_result}")
        return extracted_result

    def calculate(self, a: float, b: float, operator: str) -> float:
        logger.debug(f"Performing calculation: {a} {operator} {b}")
        if operator == '+':
            return self.add(a, b)
        elif operator == '-':
            return self.subtract(a, b)
        elif operator in ['*', 'x']:  # Handle both '*' and 'x' for multiplication
            return self.multiply(a, b)
        elif operator == '/':
            return self.divide(a, b)
        else:
            logger.error(f"Unsupported operator: {operator}")
            raise ValueError(f"Unsupported operator: {operator}")
