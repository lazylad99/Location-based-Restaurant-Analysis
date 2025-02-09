import pandas as pd
import folium
from folium.plugins import MarkerCluster
import json
import requests

def create_india_restaurant_map():
    # Read the restaurant data
    df = pd.read_csv('restaurants.csv')
    
    # Create a base map centered on India
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    try:
        # Add India state boundaries
        india_states_url = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
        states_geojson = json.loads(requests.get(india_states_url).text)
        
        folium.GeoJson(
            states_geojson,
            style_function=lambda x: {
                'fillColor': '#ffff00',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.1
            }
        ).add_to(india_map)
    except:
        print("Warning: Could not load state boundaries. Continuing without them.")
    
    # Create a marker cluster group
    marker_cluster = MarkerCluster().add_to(india_map)
    
    # Dictionary to store city-wise statistics
    city_stats = {}
    
    # Add markers for each restaurant with clustering
    for idx, row in df.iterrows():
        try:
            # Skip if latitude or longitude is missing or invalid
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
            
            # Handle missing or NaN cuisines
            cuisines = str(row['cuisines']) if pd.notna(row['cuisines']) else "Not specified"
            cuisine_list = [c.strip() for c in cuisines.split(',')] if cuisines != "Not specified" else ["Not specified"]
            
            # Update city statistics
            if row['city'] not in city_stats:
                city_stats[row['city']] = {
                    'count': 0,
                    'total_rating': 0,
                    'avg_cost': 0,
                    'cuisines': set()
                }
            
            # Handle potential NaN values in numerical fields
            rating = float(row['aggregate_rating']) if pd.notna(row['aggregate_rating']) else 0
            cost = float(row['average_cost_for_two']) if pd.notna(row['average_cost_for_two']) else 0
            
            city_stats[row['city']]['count'] += 1
            city_stats[row['city']]['total_rating'] += rating
            city_stats[row['city']]['avg_cost'] += cost
            city_stats[row['city']]['cuisines'].update(cuisine_list)
            
            # Create popup content with error handling
            popup_content = f"""
                <b>{row['name']}</b><br>
                City: {row['city']}<br>
                Rating: {rating}/5<br>
                Cost for Two: ₹{cost}<br>
                Cuisines: {cuisines}<br>
                Type: {row['type'] if pd.notna(row['type']) else 'Not specified'}
            """
            
            # Add marker to cluster
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
            
        except Exception as e:
            print(f"Warning: Skipping row {idx} due to error: {str(e)}")
            continue
    
    # Calculate and display city-wise statistics
    city_summary = {}
    for city, stats in city_stats.items():
        if stats['count'] > 0:  # Avoid division by zero
            city_summary[city] = {
                'Restaurant Count': stats['count'],
                'Average Rating': round(stats['total_rating'] / stats['count'], 2),
                'Average Cost for Two': round(stats['avg_cost'] / stats['count'], 2),
                'Unique Cuisines': len(stats['cuisines'])
            }
    
    # Convert to DataFrame and sort
    city_summary_df = pd.DataFrame.from_dict(city_summary, orient='index')
    city_summary_df = city_summary_df.sort_values('Restaurant Count', ascending=False)
    
    # Save city statistics to CSV
    city_summary_df.to_csv('city_statistics.csv')
    
    # Add a title to the map
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
    
    # Save the map
    india_map.save('india_restaurant_clusters.html')
    
    return city_summary_df

def print_city_insights(city_summary):
    print("\n=== City-wise Restaurant Analysis ===")
    
    print("\nTop 10 Cities by Restaurant Count:")
    print(city_summary['Restaurant Count'].head(10))
    
    print("\nTop 10 Cities by Average Rating:")
    print(city_summary.sort_values('Average Rating', ascending=False)['Average Rating'].head(10))
    
    print("\nTop 10 Cities by Average Cost:")
    print(city_summary.sort_values('Average Cost for Two', ascending=False)['Average Cost for Two'].head(10))
    
    print("\nTop 10 Cities by Cuisine Diversity:")
    print(city_summary.sort_values('Unique Cuisines', ascending=False)['Unique Cuisines'].head(10))

# Run the analysis
try:
    print("Starting analysis...")
    city_summary = create_india_restaurant_map()
    print_city_insights(city_summary)
    print("\nAnalysis complete! Check 'india_restaurant_clusters.html' for the interactive map")
    print("City statistics have been saved to 'city_statistics.csv'")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    raise  # Re-raise the exception to see the full traceback