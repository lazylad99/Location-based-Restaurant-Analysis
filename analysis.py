import pandas as pd
import folium
from folium.plugins import MarkerCluster
import json
import requests
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class CityStats:
    count: int = 0
    total_rating: float = 0
    avg_cost: float = 0
    cuisines: set = None
    
    def __post_init__(self):
        if self.cuisines is None:
            self.cuisines = set()

class RestaurantAnalyzer:
    def __init__(self, data_path: str):
        self.logger = self.setup_logging()
        self.df = self._load_data(data_path)
        self.city_stats: Dict[str, CityStats] = {}
        
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate restaurant data."""
        try:
            df = pd.read_csv(data_path)
            required_columns = ['name', 'city', 'latitude', 'longitude', 
                              'aggregate_rating', 'average_cost_for_two', 'cuisines']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def create_map(self, output_path: str = 'india_restaurant_clusters.html') -> None:
        """Create an interactive map with restaurant clusters."""
        india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
        self._add_state_boundaries(india_map)
        marker_cluster = MarkerCluster().add_to(india_map)
        
        for idx, row in self.df.iterrows():
            try:
                if self._is_valid_location(row):
                    self._process_restaurant(row, marker_cluster)
            except Exception as e:
                self.logger.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        self._add_map_title(india_map)
        india_map.save(output_path)
        self.logger.info(f"Map saved to {output_path}")
    
    def _add_state_boundaries(self, india_map: folium.Map) -> None:
        """Add state boundaries to the map."""
        try:
            india_states_url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
            response = requests.get(india_states_url, timeout=10)
            response.raise_for_status()
            states_geojson = response.json()
            
            folium.GeoJson(
                states_geojson,
                style_function=lambda x: {
                    'fillColor': '#ffff00',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            ).add_to(india_map)
        except Exception as e:
            self.logger.warning(f"Could not load state boundaries: {str(e)}")
    
    def _is_valid_location(self, row: pd.Series) -> bool:
        """Check if the location data is valid."""
        return not (pd.isna(row['latitude']) or pd.isna(row['longitude']))
    
    def _process_restaurant(self, row: pd.Series, marker_cluster: MarkerCluster) -> None:
        """Process a single restaurant row and update statistics."""
        cuisines = self._get_cuisines(row)
        self._update_city_stats(row, cuisines)
        self._add_restaurant_marker(row, cuisines, marker_cluster)
    
    def _get_cuisines(self, row: pd.Series) -> list:
        """Extract and process cuisine information."""
        cuisines = str(row['cuisines']) if pd.notna(row['cuisines']) else "Not specified"
        return [c.strip() for c in cuisines.split(',')] if cuisines != "Not specified" else ["Not specified"]
    
    def _update_city_stats(self, row: pd.Series, cuisines: list) -> None:
        """Update statistics for a city."""
        city = row['city']
        if city not in self.city_stats:
            self.city_stats[city] = CityStats()
        
        stats = self.city_stats[city]
        stats.count += 1
        stats.total_rating += float(row['aggregate_rating']) if pd.notna(row['aggregate_rating']) else 0
        stats.avg_cost += float(row['average_cost_for_two']) if pd.notna(row['average_cost_for_two']) else 0
        stats.cuisines.update(cuisines)
    
    def _add_restaurant_marker(self, row: pd.Series, cuisines: list, marker_cluster: MarkerCluster) -> None:
        """Add a restaurant marker to the map."""
        popup_content = self._create_popup_content(row, cuisines)
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
    
    def _create_popup_content(self, row: pd.Series, cuisines: list) -> str:
        """Create HTML content for restaurant popup."""
        return f"""
            <b>{row['name']}</b><br>
            City: {row['city']}<br>
            Rating: {row['aggregate_rating'] if pd.notna(row['aggregate_rating']) else 'N/A'}/5<br>
            Cost for Two: ₹{row['average_cost_for_two'] if pd.notna(row['average_cost_for_two']) else 'N/A'}<br>
            Cuisines: {', '.join(cuisines)}<br>
            Type: {row['type'] if pd.notna(row['type']) else 'Not specified'}
        """
    
    @staticmethod
    def _add_map_title(india_map: folium.Map) -> None:
        """Add a title to the map."""
        title_html = '''
            <div style="position: fixed; 
                        top: 10px; left: 50px; width: 300px; height: 30px; 
                        background-color: white; border-radius: 5px;
                        z-index: 9999; font-size: 16px; font-weight: bold;
                        padding: 5px;">
                Restaurant Clusters Across India
            </div>
        '''
        india_map.get_root().html.add_child(folium.Element(title_html))
    
    def generate_city_summary(self) -> pd.DataFrame:
        """Generate summary statistics for all cities."""
        city_summary = {}
        for city, stats in self.city_stats.items():
            if stats.count > 0:
                city_summary[city] = {
                    'Restaurant Count': stats.count,
                    'Average Rating': round(stats.total_rating / stats.count, 2),
                    'Average Cost for Two': round(stats.avg_cost / stats.count, 2),
                    'Unique Cuisines': len(stats.cuisines)
                }
        
        summary_df = pd.DataFrame.from_dict(city_summary, orient='index')
        return summary_df.sort_values('Restaurant Count', ascending=False)
    
    def save_city_statistics(self, output_path: str = 'city_statistics.csv') -> None:
        """Save city statistics to CSV."""
        summary_df = self.generate_city_summary()
        summary_df.to_csv(output_path)
        self.logger.info(f"City statistics saved to {output_path}")
    
    def print_insights(self) -> None:
        """Print key insights about the cities."""
        summary_df = self.generate_city_summary()
        
        print("\n=== City-wise Restaurant Analysis ===")
        
        print("\nTop 10 Cities by Restaurant Count:")
        print(summary_df['Restaurant Count'].head(10))
        
        print("\nTop 10 Cities by Average Rating:")
        print(summary_df.sort_values('Average Rating', ascending=False)['Average Rating'].head(10))
        
        print("\nTop 10 Cities by Average Cost:")
        print(summary_df.sort_values('Average Cost for Two', ascending=False)['Average Cost for Two'].head(10))
        
        print("\nTop 10 Cities by Cuisine Diversity:")
        print(summary_df.sort_values('Unique Cuisines', ascending=False)['Unique Cuisines'].head(10))

def main():
    try:
        analyzer = RestaurantAnalyzer('restaurants.csv')
        analyzer.create_map()
        analyzer.save_city_statistics()
        analyzer.print_insights()
        print("\nAnalysis complete! Check 'india_restaurant_clusters.html' for the interactive map")
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()