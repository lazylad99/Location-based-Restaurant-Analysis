# India Restaurant Analysis Project 🍽️

![Restaurant Clusters Map](https://raw.githubusercontent.com/yourusername/restaurant-analysis/main/screenshots/map_preview.png)

## Overview 📊

This project analyzes restaurant data across India, creating interactive visualizations and statistical insights about restaurant distributions, ratings, cuisines, and pricing patterns. The analysis includes geographic clustering, city-wise comparisons, and detailed statistical breakdowns.

## Features 🌟

- **Interactive Map Visualization**
  - Restaurant locations plotted on India map
  - Cluster visualization for dense areas
  - Click-able markers with restaurant details
  - State boundary overlays

- **Statistical Analysis**
  - City-wise restaurant distribution
  - Rating analysis across regions
  - Price range distribution
  - Cuisine diversity metrics

- **Data Insights**
  - Top cities by restaurant count
  - Average ratings comparison
  - Cost analysis by region
  - Cuisine type distribution

## Requirements 📋

```python
pandas==1.5.3
folium==0.14.0
requests==2.28.2
```

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/restaurant-analysis.git
cd restaurant-analysis
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Place your restaurant data CSV file in the project directory
2. Run the analysis script:
```bash
python analysis.py
```

3. Check the generated outputs:
   - `india_restaurant_clusters.html`: Interactive map
   - `city_statistics.csv`: Detailed city-wise analysis

## Data Format 📝

The input CSV should contain the following columns:
```
res_id, name, type, address, city, locality, latitude, longitude, 
cuisines, timings, average_cost_for_two, price_range, highlights, 
aggregate_rating, votes, photo_count, opentable_support, delivery, 
state, area
```

## Output Examples 📊

### Interactive Map
The generated map includes:
- Restaurant markers clustered by proximity
- Popup information for each restaurant
- Color-coded state boundaries
- Zoom functionality for detailed exploration

### Statistical Insights
```
=== City-wise Restaurant Analysis ===
Top 10 Cities by Restaurant Count:
Mumbai     : 1245
Delhi      : 1132
Bangalore  : 987
...

Top 10 Cities by Average Rating:
Chandigarh : 4.3
Jaipur     : 4.2
Pune       : 4.1
...
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments 👏

- Restaurant data sourced from [source name]
- India GeoJSON data from [Subhash9325's repository](https://github.com/Subhash9325/GeoJson-Data-of-Indian-States)

## Contact 📧

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/restaurant-analysis](https://github.com/yourusername/restaurant-analysis)
