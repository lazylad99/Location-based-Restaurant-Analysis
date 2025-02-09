import pandas as pd
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
import json
import requests
import logging
from pathlib import Path
import numpy as np
from scipy import stats

@dataclass
class CityStats:
    count: int = 0
    total_rating: float = 0
    avg_cost: float = 0
    cuisines: dict = None
    price_ranges: list = None
    
    def __post_init__(self):
        if self.cuisines is None:
            self.cuisines = {}
        if self.price_ranges is None:
            self.price_ranges = []

class RestaurantAnalyzer:
    def __init__(self, data_path: str):
        self.logger = self.setup_logging()
        self.df = self._load_and_clean_data(data_path)
        self.city_stats: Dict[str, CityStats] = {}
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_and_clean_data(self, data_path: str) -> pd.DataFrame:
        """Load, validate, and clean restaurant data."""
        try:
            df = pd.read_csv(data_path)
            required_columns = ['name', 'city', 'latitude', 'longitude', 
                              'aggregate_rating', 'average_cost_for_two', 'cuisines']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean and preprocess data
            df['cuisines'] = df['cuisines'].fillna('Not specified')
            df['aggregate_rating'] = pd.to_numeric(df['aggregate_rating'], errors='coerce')
            df['average_cost_for_two'] = pd.to_numeric(df['average_cost_for_two'], errors='coerce')
            
            # Add price category
            df['price_category'] = pd.qcut(df['average_cost_for_two'].fillna(df['average_cost_for_two'].median()),
                                         q=4, labels=['Budget', 'Moderate', 'High-End', 'Luxury'])
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def create_visualizations(self):
        """Generate all visualizations."""
        self._create_interactive_map()
        self._create_restaurant_density_analysis()
        self._create_rating_analysis()
        self._create_cuisine_analysis()
        self._create_price_analysis()
        self._create_correlation_analysis()
        
    def _add_restaurant_marker(self, row: pd.Series, marker_cluster: folium.plugins.MarkerCluster) -> None:
        """Add a restaurant marker to the map with popup information."""
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            popup_content = self._create_popup_content(row)
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(
                    color=self._get_marker_color(row['aggregate_rating']),
                    icon='info-sign'
                )
            ).add_to(marker_cluster)

    def _create_popup_content(self, row: pd.Series) -> str:
        """Create HTML content for restaurant popup."""
        rating = row['aggregate_rating'] if pd.notna(row['aggregate_rating']) else 'N/A'
        cost = row['average_cost_for_two'] if pd.notna(row['average_cost_for_two']) else 'N/A'
        cuisines = row['cuisines'] if pd.notna(row['cuisines']) else 'Not specified'
        
        return f"""
            <div style="font-family: Arial, sans-serif; width: 200px;">
                <h4 style="margin-bottom: 5px;">{row['name']}</h4>
                <p style="margin: 2px 0;"><b>City:</b> {row['city']}</p>
                <p style="margin: 2px 0;"><b>Rating:</b> {rating}/5</p>
                <p style="margin: 2px 0;"><b>Cost for Two:</b> ₹{cost}</p>
                <p style="margin: 2px 0;"><b>Cuisines:</b> {cuisines}</p>
                <p style="margin: 2px 0;"><b>Category:</b> {row['price_category']}</p>
            </div>
        """

    def _get_marker_color(self, rating: float) -> str:
        """Return marker color based on restaurant rating."""
        if pd.isna(rating):
            return 'gray'
        elif rating >= 4.5:
            return 'darkgreen'
        elif rating >= 4.0:
            return 'green'
        elif rating >= 3.5:
            return 'orange'
        elif rating >= 3.0:
            return 'lightred'
        else:
            return 'red'

    def _create_interactive_map(self):
        """Create interactive map with multiple layers."""
        # Base map
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        
        # Add different layers
        marker_cluster = MarkerCluster(name="Restaurant Clusters").add_to(m)
        heat_layer = folium.FeatureGroup(name="Density Heatmap")
        
        # Add restaurants to layers
        heat_data = []
        for _, row in self.df.iterrows():
            if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                # Add to cluster
                self._add_restaurant_marker(row, marker_cluster)
                
                # Add to heat data
                weight = 1 + (row['aggregate_rating'] if pd.notna(row['aggregate_rating']) else 0) / 5
                heat_data.append([row['latitude'], row['longitude'], weight])
        
        # Create heatmap
        HeatMap(heat_data).add_to(heat_layer)
        heat_layer.add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; right: 50px; 
                    background-color: white; padding: 10px; border-radius: 5px;
                    z-index: 1000;">
            <h4>Rating Colors</h4>
            <p><i class="fa fa-map-marker" style="color: darkgreen;"></i> 4.5+</p>
            <p><i class="fa fa-map-marker" style="color: green;"></i> 4.0-4.4</p>
            <p><i class="fa fa-map-marker" style="color: orange;"></i> 3.5-3.9</p>
            <p><i class="fa fa-map-marker" style="color: lightred;"></i> 3.0-3.4</p>
            <p><i class="fa fa-map-marker" style="color: red;"></i> < 3.0</p>
            <p><i class="fa fa-map-marker" style="color: gray;"></i> No rating</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(self.output_dir / 'interactive_map.html')




    def _create_restaurant_density_analysis(self):
        """Create visualizations for restaurant density analysis."""
        # City-wise restaurant count
        city_counts = self.df['city'].value_counts().reset_index()
        city_counts.columns = ['city', 'count']
        
        fig = px.bar(city_counts,
                    x='city',
                    y='count',
                    title='Restaurant Count by City',
                    labels={'city': 'City', 'count': 'Number of Restaurants'})
        
        fig.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="City",
            yaxis_title="Number of Restaurants",
            showlegend=False,
            plot_bgcolor='white'
        )
        
        fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)
        fig.write_html(self.output_dir / 'restaurant_density.html')
        
        # Create density map using updated function
        fig = px.density_map(self.df,
                            lat='latitude',
                            lon='longitude',
                            z='aggregate_rating',
                            radius=10,
                            center=dict(lat=20.5937, lon=78.9629),
                            zoom=4,
                            title='Restaurant Density Map')
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        
        fig.write_html(self.output_dir / 'density_map.html')


    def _create_rating_analysis(self):
        """Create visualizations for rating analysis."""
        # Average ratings by city
        city_ratings = self.df.groupby('city')['aggregate_rating'].agg(['mean', 'count']).reset_index()
        city_ratings = city_ratings[city_ratings['count'] > 10]  # Filter cities with sufficient data
        
        fig = px.bar(city_ratings.sort_values('mean', ascending=False),
                    x='city', y='mean',
                    title='Average Restaurant Ratings by City',
                    labels={'mean': 'Average Rating', 'city': 'City'})
        fig.write_html(self.output_dir / 'city_ratings.html')
        
        # Rating distribution
        fig = ff.create_distplot([self.df['aggregate_rating'].dropna()],
                               ['Rating Distribution'],
                               bin_size=0.25)
        fig.update_layout(title='Distribution of Restaurant Ratings')
        fig.write_html(self.output_dir / 'rating_distribution.html')

    def _create_cuisine_analysis(self):
        """Create visualizations for cuisine analysis."""
        # Process cuisine data
        cuisine_counts = {}
        for cuisines in self.df['cuisines'].str.split(','):
            for cuisine in cuisines:
                cuisine = cuisine.strip()
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400,
                            background_color='white').generate_from_frequencies(cuisine_counts)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(self.output_dir / 'cuisine_wordcloud.png')
        plt.close()
        
        # Top cuisines by city
        fig = px.bar(pd.DataFrame(list(cuisine_counts.items()),
                                columns=['Cuisine', 'Count']).sort_values('Count', ascending=False).head(20),
                    x='Cuisine', y='Count',
                    title='Most Popular Cuisines')
        fig.write_html(self.output_dir / 'top_cuisines.html')

    def _create_price_analysis(self):
        """Create visualizations for price analysis."""
        try:
            # Price distribution by city
            fig = px.box(self.df,
                        x='city',
                        y='average_cost_for_two',
                        title='Restaurant Price Distribution by City')
            fig.update_layout(xaxis_tickangle=-45)
            fig.write_html(self.output_dir / 'price_distribution.html')
            
            # Price category distribution
            fig = px.pie(self.df,
                        names='price_category',
                        title='Distribution of Restaurant Price Categories')
            fig.write_html(self.output_dir / 'price_categories.html')
            
            # Average price by cuisine
            cuisine_prices = []
            for _, row in self.df.iterrows():
                if pd.notna(row['cuisines']) and pd.notna(row['average_cost_for_two']):
                    for cuisine in row['cuisines'].split(','):
                        cuisine = cuisine.strip()
                        if cuisine:  # Skip empty cuisines
                            cuisine_prices.append({
                                'cuisine': cuisine,
                                'price': row['average_cost_for_two']
                            })
            
            # Create DataFrame and calculate statistics
            cuisine_price_df = pd.DataFrame(cuisine_prices)
            cuisine_stats = cuisine_price_df.groupby('cuisine').agg({
                'price': ['mean', 'count']
            }).reset_index()
            
            # Flatten column names
            cuisine_stats.columns = ['cuisine', 'mean_price', 'count']
            
            # Filter for cuisines with sufficient data
            min_restaurants = 5  # Minimum number of restaurants per cuisine
            filtered_stats = cuisine_stats[cuisine_stats['count'] >= min_restaurants]
            
            # Sort by mean price and get top 20
            top_20_cuisines = filtered_stats.sort_values('mean_price', ascending=False).head(20)
            
            # Create bar chart
            fig = px.bar(top_20_cuisines,
                        x='cuisine',
                        y='mean_price',
                        title='Average Price by Cuisine Type (Top 20)',
                        labels={
                            'cuisine': 'Cuisine',
                            'mean_price': 'Average Cost for Two (₹)'
                        })
            
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                plot_bgcolor='white'
            )
            
            fig.write_html(self.output_dir / 'cuisine_prices.html')
            
            # Additional visualization: Price ranges
            price_ranges = pd.qcut(self.df['average_cost_for_two'].dropna(), 
                                q=5, 
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            fig = px.histogram(x=price_ranges,
                            title='Distribution of Restaurant Price Ranges',
                            labels={'x': 'Price Range', 'count': 'Number of Restaurants'},
                            color_discrete_sequence=['indianred'])
            
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='white'
            )
            
            fig.write_html(self.output_dir / 'price_ranges.html')
            
        except Exception as e:
            self.logger.error(f"Error in price analysis: {str(e)}")
            print(f"Warning: Could not complete price analysis due to error: {str(e)}")

    def _create_correlation_analysis(self):
        """Create correlation analysis visualizations."""
        try:
            import statsmodels.api as sm
            has_statsmodels = True
        except ImportError:
            has_statsmodels = False
            self.logger.warning("statsmodels not installed. Will create correlation plots without trend lines.")

        # Correlation between rating and price
        fig = px.scatter(self.df,
                        x='average_cost_for_two',
                        y='aggregate_rating',
                        trendline="lowess" if has_statsmodels else None,
                        title='Correlation between Price and Rating',
                        labels={
                            'average_cost_for_two': 'Average Cost for Two (₹)',
                            'aggregate_rating': 'Rating'
                        })
        
        # Calculate correlation coefficient
        corr_coef = self.df['average_cost_for_two'].corr(self.df['aggregate_rating'])
        
        fig.add_annotation(
            text=f'Correlation: {corr_coef:.2f}',
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            showlegend=True
        )
        
        fig.write_html(self.output_dir / 'price_rating_correlation.html')
        
        # Restaurant density vs average rating by city
        city_stats = self.df.groupby('city').agg({
            'aggregate_rating': 'mean',
            'name': 'count'
        }).reset_index()
        
        city_stats.columns = ['city', 'avg_rating', 'restaurant_count']
        
        # Only include cities with sufficient data
        min_restaurants = 5
        filtered_city_stats = city_stats[city_stats['restaurant_count'] >= min_restaurants]
        
        fig = px.scatter(filtered_city_stats,
                        x='restaurant_count',
                        y='avg_rating',
                        text='city',
                        title='Restaurant Density vs Average Rating by City',
                        labels={
                            'restaurant_count': 'Number of Restaurants',
                            'avg_rating': 'Average Rating'
                        })
        
        # Calculate correlation for density vs rating
        density_rating_corr = filtered_city_stats['restaurant_count'].corr(filtered_city_stats['avg_rating'])
        
        fig.add_annotation(
            text=f'Correlation: {density_rating_corr:.2f}',
            xref='paper',
            yref='paper',
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
        
        fig.update_traces(
            textposition='top center',
            marker=dict(size=10)
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            showlegend=False
        )
        
        fig.write_html(self.output_dir / 'density_rating_correlation.html')
        
        # Create correlation matrix
        numeric_cols = ['aggregate_rating', 'average_cost_for_two']
        corr_matrix = self.df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu',
                        title='Correlation Matrix of Numeric Variables')
        
        # Add correlation values as text
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    text=f"{corr_matrix.iloc[i, j]:.2f}",
                    x=j,
                    y=i,
                    showarrow=False,
                    font=dict(color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
                )
        
        fig.write_html(self.output_dir / 'correlation_matrix.html')
        
        
        

    def generate_insights_report(self) -> str:
        """Generate a comprehensive insights report."""
        report = []
        report.append("# Restaurant Analysis Insights Report\n")
        
        # City-level insights
        top_cities = self.df['city'].value_counts().head(5)
        report.append("## Top Restaurant Cities")
        report.append("Cities with the most restaurants:")
        for city, count in top_cities.items():
            report.append(f"- {city}: {count} restaurants")
        
        # Rating insights
        report.append("\n## Rating Analysis")
        top_rated_cities = self.df.groupby('city')['aggregate_rating'].mean().sort_values(ascending=False).head(5)
        report.append("Cities with the highest average ratings:")
        for city, rating in top_rated_cities.items():
            report.append(f"- {city}: {rating:.2f}/5")
        
        # Cuisine insights
        report.append("\n## Cuisine Analysis")
        cuisine_counts = {}
        for cuisines in self.df['cuisines'].str.split(','):
            for cuisine in cuisines:
                cuisine = cuisine.strip()
                cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
        
        top_cuisines = dict(sorted(cuisine_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        report.append("Most popular cuisines:")
        for cuisine, count in top_cuisines.items():
            report.append(f"- {cuisine}: {count} restaurants")
        
        # Price insights
        report.append("\n## Price Analysis")
        avg_price_by_city = self.df.groupby('city')['average_cost_for_two'].mean().sort_values(ascending=False).head(5)
        report.append("Cities with the highest average cost for two:")
        for city, price in avg_price_by_city.items():
            report.append(f"- {city}: ₹{price:.2f}")
        
        # Correlation insights
        price_rating_corr = self.df['average_cost_for_two'].corr(self.df['aggregate_rating'])
        report.append(f"\n## Correlations")
        report.append(f"Correlation between price and rating: {price_rating_corr:.2f}")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / 'insights_report.md', 'w') as f:
            f.write(report_text)
        
        return report_text

def main():
    try:
        analyzer = RestaurantAnalyzer('restaurants.csv')
        analyzer.create_visualizations()
        insights = analyzer.generate_insights_report()
        print("Analysis complete! Check the 'output' directory for all visualizations and reports.")
        print("\nKey Insights:\n")
        print(insights)
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()