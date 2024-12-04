import numpy as np
import pandas as pd
from typing import List, Dict, Union

class DCFModel:
    def __init__(self):
        self.risk_free_rate = None
        self.market_return = None
        self.beta = None
        self.cost_of_debt = None
        self.tax_rate = None
        self.debt_ratio = None
        self.terminal_growth = None
        
    def calculate_beta(self, stock_returns: List[float], market_returns: List[float]) -> float:
        """
        Calculate 5-year weekly beta using regression
        """
        stock_returns = np.array(stock_returns)
        market_returns = np.array(market_returns)
        
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance
    
    def calculate_cost_of_equity(self) -> float:
        """
        Calculate cost of equity using CAPM
        Re = Rf + β(Rm - Rf)
        """
        equity_risk_premium = self.market_return - self.risk_free_rate
        return self.risk_free_rate + (self.beta * equity_risk_premium)
    
    def calculate_wacc(self) -> float:
        """
        Calculate WACC
        WACC = (E/V × Re) + (D/V × Rd × (1-T))
        """
        equity_ratio = 1 - self.debt_ratio
        cost_of_equity = self.calculate_cost_of_equity()
        
        return (equity_ratio * cost_of_equity + 
                self.debt_ratio * self.cost_of_debt * (1 - self.tax_rate))
    
    def calculate_present_value(self, cash_flows: List[float], periods: List[int]) -> float:
        """
        Calculate present value of cash flows
        """
        wacc = self.calculate_wacc()
        present_values = []
        
        for cf, period in zip(cash_flows, periods):
            present_value = cf / ((1 + wacc) ** period)
            present_values.append(present_value)
            
        return sum(present_values)
    
    def calculate_terminal_value(self, final_cash_flow: float) -> float:
        """
        Calculate terminal value using Gordon Growth model
        TV = FCF(1+g)/(WACC-g)
        """
        wacc = self.calculate_wacc()
        terminal_value = (final_cash_flow * (1 + self.terminal_growth) / 
                         (wacc - self.terminal_growth))
        
        return terminal_value
    
    def perform_dcf_valuation(self, 
                            projected_cash_flows: List[float],
                            stock_returns: List[float],
                            market_returns: List[float],
                            risk_free_rate: float,
                            market_return: float,
                            cost_of_debt: float,
                            tax_rate: float,
                            debt_ratio: float,
                            terminal_growth: float) -> Dict[str, Union[float, dict]]:
        """
        Perform complete DCF valuation
        """
        # Set parameters
        self.risk_free_rate = risk_free_rate
        self.market_return = market_return
        self.cost_of_debt = cost_of_debt
        self.tax_rate = tax_rate
        self.debt_ratio = debt_ratio
        self.terminal_growth = terminal_growth
        
        # Calculate beta
        self.beta = self.calculate_beta(stock_returns, market_returns)
        
        # Calculate components
        wacc = self.calculate_wacc()
        cost_of_equity = self.calculate_cost_of_equity()
        
        # Calculate present value of explicit forecast period
        periods = list(range(1, len(projected_cash_flows) + 1))
        pv_explicit = self.calculate_present_value(projected_cash_flows, periods)
        
        # Calculate terminal value
        terminal_value = self.calculate_terminal_value(projected_cash_flows[-1])
        pv_terminal = terminal_value / ((1 + wacc) ** len(projected_cash_flows))
        
        # Total value
        total_value = pv_explicit + pv_terminal
        
        # Return detailed results
        return {
            'enterprise_value': total_value,
            'present_value_explicit': pv_explicit,
            'present_value_terminal': pv_terminal,
            'terminal_value': terminal_value,
            'components': {
                'wacc': wacc,
                'cost_of_equity': cost_of_equity,
                'beta': self.beta
            }
        }

# Example usage
def example_dcf():
    # Sample data
    projected_cash_flows = [100, 110, 120, 130, 140]  # 5-year projection
    stock_returns = [0.05, 0.03, 0.04, 0.06, 0.02]    # Weekly returns for beta
    market_returns = [0.03, 0.02, 0.04, 0.05, 0.01]   # Market returns for beta
    
    # Parameters
    risk_free_rate = 0.04      # 4% risk-free rate
    market_return = 0.10       # 10% market return
    cost_of_debt = 0.06        # 6% cost of debt
    tax_rate = 0.25           # 25% tax rate
    debt_ratio = 0.30         # 30% debt in capital structure
    terminal_growth = 0.02    # 2% terminal growth
    
    # Create and run model
    model = DCFModel()
    valuation = model.perform_dcf_valuation(
        projected_cash_flows=projected_cash_flows,
        stock_returns=stock_returns,
        market_returns=market_returns,
        risk_free_rate=risk_free_rate,
        market_return=market_return,
        cost_of_debt=cost_of_debt,
        tax_rate=tax_rate,
        debt_ratio=debt_ratio,
        terminal_growth=terminal_growth
    )
    
    return pd.DataFrame([valuation])

# Run example
if __name__ == "__main__":
    result = example_dcf()
    print("\nDCF Valuation Results:")
    print(result)