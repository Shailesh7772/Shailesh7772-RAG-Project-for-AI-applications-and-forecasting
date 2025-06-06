import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go

class RealEstatePredictor:
    def __init__(self):
        self.model = None
        self.region_data = None
        self.historical_data = None

    def prepare_data(self, df: pd.DataFrame, region: str) -> pd.DataFrame:
        """Prepare Zillow data for Prophet"""
        try:
            # Get date columns (assuming they start with '20')
            date_columns = [col for col in df.columns if str(col).startswith('20')]
            
            # Melt the dataframe
            df_melted = pd.melt(
                df,
                id_vars=['RegionName', 'RegionID', 'StateName'],
                value_vars=date_columns,
                var_name='ds',
                value_name='y'
            )
            
            # Convert date strings to datetime
            df_melted['ds'] = pd.to_datetime(df_melted['ds'])
            
            # Filter for selected region and sort by date
            df_region = df_melted[df_melted['RegionName'] == region].copy()
            df_region = df_region.sort_values('ds')
            
            # Store region data for later use
            self.region_data = df_region
            self.historical_data = df_region[['ds', 'y']].copy()
            
            return df_region[['ds', 'y']].copy()
            
        except Exception as e:
            raise Exception(f"Error preparing data: {str(e)}")
    
    def train(self, df: pd.DataFrame, region: str):
        """Train Prophet model"""
        try:
            # Prepare data
            prophet_df = self.prepare_data(df, region)
            
            # Initialize and train model
            self.model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            self.model.fit(prophet_df)
            return "Model trained successfully"
            
        except Exception as e:
            raise Exception(f"Error training model: {str(e)}")

    def evaluate_model(self):
        """Calculate prediction accuracy metrics"""
        try:
            if self.historical_data is None:
                raise Exception("No historical data available")

            # Get predictions for historical period
            historical_dates = self.historical_data[['ds']].copy()
            historical_forecast = self.model.predict(historical_dates)
            
            # Get actual and predicted values
            actual = self.historical_data['y'].values
            predicted = historical_forecast['yhat'].values
            
            # Calculate metrics
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            
            # Calculate training period
            date_range = pd.date_range(
                start=self.historical_data['ds'].min(),
                end=self.historical_data['ds'].max(),
                freq='M'
            )
            
            return {
                'MAPE': f"{mape:.2f}%",
                'RMSE': f"${rmse:,.2f}",
                'Data Points': len(actual),
                'Training Period': f"{len(date_range)} months"
            }
            
        except Exception as e:
            raise Exception(f"Error calculating metrics: {str(e)}")
    
    def predict(self, periods: int):
        """Generate predictions"""
        try:
            if self.model is None:
                raise Exception("Model not trained. Call train() first.")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=periods, freq='M')
            forecast = self.model.predict(future)
            
            # Create plot
            fig = go.Figure()
            
            # Historical values
            fig.add_trace(go.Scatter(
                x=self.historical_data['ds'],
                y=self.historical_data['y'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Predictions
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tail(periods),
                y=forecast['yhat'].tail(periods),
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tail(periods),
                y=forecast['yhat_upper'].tail(periods),
                fill=None,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.2)'),
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tail(periods),
                y=forecast['yhat_lower'].tail(periods),
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255,0,0,0.2)'),
                name='Lower Bound'
            ))
            
            # Update layout
            region_name = self.region_data['RegionName'].iloc[0]
            state_name = self.region_data['StateName'].iloc[0]
            
            fig.update_layout(
                title=f'Real Estate Price Prediction - {region_name}, {state_name}',
                xaxis_title='Date',
                yaxis_title='Home Value ($)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods), fig
            
        except Exception as e:
            raise Exception(f"Error making prediction: {str(e)}")