import numpy as np
import pandas as pd
from typing import List, Dict, Union
from scipy import stats

class CashFlowProjections:
    def __init__(self, historical_data: pd.DataFrame):
        self.historical_data = historical_data
        
    def growth_rate_method(self, base_fcf: float, growth_rates: List[float]) -> List[float]:
        """
        Project cash flows using specified growth rates
        
        Parameters:
        base_fcf: Last known free cash flow
        growth_rates: List of expected growth rates for each year
        """
        projected_fcf = []
        current_fcf = base_fcf
        
        for growth_rate in growth_rates:
            current_fcf *= (1 + growth_rate)
            projected_fcf.append(current_fcf)
            
        return projected_fcf
    
    def revenue_driven_method(self,
                            base_revenue: float,
                            revenue_growth: List[float],
                            fcf_margin: float,
                            margin_growth: float = 0) -> List[float]:
        """
        Project cash flows based on revenue growth and FCF margins
        
        Parameters:
        base_revenue: Last known revenue
        revenue_growth: Expected revenue growth rates
        fcf_margin: Current FCF margin (FCF/Revenue)
        margin_growth: Expected annual improvement in FCF margin
        """
        projected_fcf = []
        current_revenue = base_revenue
        current_margin = fcf_margin
        
        for growth_rate in revenue_growth:
            current_revenue *= (1 + growth_rate)
            current_margin += margin_growth
            fcf = current_revenue * current_margin
            projected_fcf.append(fcf)
            
        return projected_fcf
    
    def bottom_up_method(self,
                        revenue: float,
                        revenue_growth: List[float],
                        operating_margin: float,
                        tax_rate: float,
                        depreciation_rate: float,
                        capex_rate: float,
                        nwc_rate: float) -> List[float]:
        """
        Project cash flows using detailed operating assumptions
        
        FCF = EBIT(1-t) + Depreciation - CapEx - ΔWorking Capital
        
        Parameters:
        revenue: Base revenue
        revenue_growth: Revenue growth rates
        operating_margin: EBIT margin
        tax_rate: Effective tax rate
        depreciation_rate: Depreciation as % of revenue
        capex_rate: CapEx as % of revenue
        nwc_rate: Net working capital as % of revenue change
        """
        projected_fcf = []
        current_revenue = revenue
        prior_revenue = revenue
        
        for growth in revenue_growth:
            # Project revenue
            current_revenue *= (1 + growth)
            
            # Calculate components
            ebit = current_revenue * operating_margin
            nopat = ebit * (1 - tax_rate)
            depreciation = current_revenue * depreciation_rate
            capex = current_revenue * capex_rate
            delta_nwc = (current_revenue - prior_revenue) * nwc_rate
            
            # Calculate FCF
            fcf = nopat + depreciation - capex - delta_nwc
            projected_fcf.append(fcf)
            
            prior_revenue = current_revenue
            
        return projected_fcf
    
    def monte_carlo_simulation(self,
                             base_fcf: float,
                             num_simulations: int,
                             num_periods: int,
                             mean_growth: float,
                             growth_std: float,
                             confidence_level: float = 0.95) -> Dict[str, Union[List[float], Dict[str, List[float]]]]:
        """
        Project cash flows using Monte Carlo simulation
        
        Parameters:
        base_fcf: Starting free cash flow
        num_simulations: Number of simulation runs
        num_periods: Number of periods to project
        mean_growth: Expected mean growth rate
        growth_std: Standard deviation of growth rate
        confidence_level: Confidence level for intervals
        """
        simulations = np.zeros((num_simulations, num_periods))
        
        for sim in range(num_simulations):
            current_fcf = base_fcf
            for period in range(num_periods):
                growth_rate = np.random.normal(mean_growth, growth_std)
                current_fcf *= (1 + growth_rate)
                simulations[sim, period] = current_fcf
        
        # Calculate mean projection and confidence intervals
        mean_projection = np.mean(simulations, axis=0)
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        confidence_intervals = {
            'lower': np.percentile(simulations, lower_percentile * 100, axis=0),
            'upper': np.percentile(simulations, upper_percentile * 100, axis=0)
        }
        
        return {
            'mean_projection': mean_projection.tolist(),
            'confidence_intervals': confidence_intervals
        }
    
    def regression_based_method(self, 
                              dependent_vars: List[str],
                              projection_periods: int) -> List[float]:
        """
        Project cash flows using multiple regression analysis
        
        Parameters:
        dependent_vars: List of variable names that predict FCF
        projection_periods: Number of periods to project
        """
        # Prepare data
        X = self.historical_data[dependent_vars]
        y = self.historical_data['fcf']
        
        # Fit regression model
        model = stats.LinearRegression()
        model.fit(X, y)
        
        # Project variables (simplified - assumes linear trends)
        future_X = pd.DataFrame()
        for var in dependent_vars:
            trend = np.polyfit(range(len(X)), X[var], 1)
            future_values = np.polyval(trend, range(len(X), len(X) + projection_periods))
            future_X[var] = future_values
        
        # Generate projections
        projected_fcf = model.predict(future_X)
        
        return projected_fcf.tolist()

# Example usage
def example_projections():
    # Sample historical data
    historical_data = pd.DataFrame({
        'year': range(2019, 2024),
        'revenue': [1000, 1100, 1210, 1331, 1464],
        'fcf': [100, 110, 121, 133, 146],
        'gdp_growth': [0.02, 0.018, 0.022, 0.019, 0.021],
        'industry_growth': [0.03, 0.028, 0.032, 0.029, 0.031]
    })
    
    projector = CashFlowProjections(historical_data)
    
    # Example 1: Simple growth rate method
    growth_rates = [0.05, 0.05, 0.04, 0.04, 0.03]
    simple_projection = projector.growth_rate_method(146, growth_rates)
    
    # Example 2: Revenue-driven method
    revenue_growth = [0.05, 0.05, 0.04, 0.04, 0.03]
    revenue_based = projector.revenue_driven_method(1464, revenue_growth, 0.1)
    
    # Example 3: Bottom-up method
    bottom_up = projector.bottom_up_method(
        revenue=1464,
        revenue_growth=[0.05, 0.05, 0.04, 0.04, 0.03],
        operating_margin=0.15,
        tax_rate=0.25,
        depreciation_rate=0.05,
        capex_rate=0.07,
        nwc_rate=0.1
    )
    
    # Example 4: Monte Carlo simulation
    monte_carlo = projector.monte_carlo_simulation(
        base_fcf=146,
        num_simulations=1000,
        num_periods=5,
        mean_growth=0.05,
        growth_std=0.02
    )
    
    return pd.DataFrame({
        'Simple Growth': simple_projection,
        'Revenue Based': revenue_based,
        'Bottom Up': bottom_up,
        'Monte Carlo (Mean)': monte_carlo['mean_projection']
    })

if __name__ == "__main__":
    results = example_projections()
    print("\nCash Flow Projections by Different Methods:")
    print(results)