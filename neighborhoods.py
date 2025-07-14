#!/usr/bin/env python3
"""
Santa Monica Neighborhood Organization Malapportionment Analysis
Analyzes demographic disparities across Santa Monica's seven official neighborhood organizations.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from census import Census
from us import states
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import requests
import zipfile
import os
from pathlib import Path

# Configuration
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

class SantaMonicaNeighborhoodAnalysis:
    def __init__(self, census_api_key):
        self.census = Census(census_api_key)
        self.ca_fips = states.CA.fips
        
    def download_neighborhood_boundaries(self):
        """Load neighborhood organization boundaries from local GeoJSON file."""
        print("Loading neighborhood organization boundaries from local GeoJSON...")
        
        geojson_path = DATA_DIR / "Neighborhood_Organizations.geojson"
        
        if not geojson_path.exists():
            raise FileNotFoundError(
                f"Neighborhood_Organizations.geojson not found at {geojson_path}. "
                "Please ensure the file is in the 'data' directory."
            )
            
        neighborhoods = gpd.read_file(geojson_path)
        # Filter out null neighborhood names
        neighborhoods = neighborhoods[neighborhoods['neighbor'].notna()]
        print(f"Loaded {len(neighborhoods)} neighborhood organizations from GeoJSON")
        return neighborhoods
    
    def download_santa_monica_boundary(self):
        """Download Santa Monica city boundary to properly filter census data."""
        print("Downloading Santa Monica city boundary...")
        
        places_url = "https://www2.census.gov/geo/tiger/TIGER2020/PLACE/tl_2020_06_place.zip"
        
        response = requests.get(places_url)
        zip_path = DATA_DIR / "places.zip"
        
        with open(zip_path, 'wb') as f:
            f.write(response.content)
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR / "places")
        
        places = gpd.read_file(DATA_DIR / "places" / "tl_2020_06_place.shp")
        santa_monica = places[places['NAME'] == 'Santa Monica']
        
        if len(santa_monica) == 0:
            raise ValueError("Santa Monica boundary not found in places file")
            
        print("Santa Monica boundary loaded successfully")
        return santa_monica.iloc[0:1]  # Return as GeoDataFrame
    
    def download_census_tracts(self):
        """Download census tract boundaries for geographic analysis."""
        print("Downloading census tract boundaries...")
        
        tracts_url = "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_06_tract.zip"
        
        zip_path = DATA_DIR / "census_tracts.zip"
        
        if not zip_path.exists():
            print(f"Downloading from: {tracts_url}")
            response = requests.get(tracts_url)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download: HTTP {response.status_code}")
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {len(response.content)} bytes")
        else:
            print("Using existing census tracts file")
            
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
            raise Exception(f"Downloaded file is not a valid zip file. Please try again.")
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR / "census_tracts")
        
        tracts = gpd.read_file(DATA_DIR / "census_tracts" / "tl_2020_06_tract.shp")
        
        # Filter to LA County tracts only
        la_county_tracts = tracts[tracts['COUNTYFP'] == '037']
        
        print(f"Loaded {len(la_county_tracts)} LA County census tracts")
        return la_county_tracts
    
    def download_census_blocks(self, santa_monica_boundary):
        """Download and filter census blocks for Santa Monica area."""
        print("Downloading census blocks...")
        
        blocks_url = "https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_06_tabblock20.zip"
        
        zip_path = DATA_DIR / "census_blocks.zip"
        
        # Check if file already exists
        if not zip_path.exists():
            print(f"Downloading from: {blocks_url}")
            response = requests.get(blocks_url)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download: HTTP {response.status_code}")
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {len(response.content)} bytes")
        else:
            print("Using existing census blocks file")
            
        # Verify it's a valid zip file
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink()  # Delete corrupted file
            raise Exception(f"Downloaded file is not a valid zip file. Please try again.")
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR / "census_blocks")
        
        blocks = gpd.read_file(DATA_DIR / "census_blocks" / "tl_2020_06_tabblock20.shp")
        
        # Convert to same CRS for spatial operations
        santa_monica_boundary = santa_monica_boundary.to_crs(blocks.crs)
        
        # Filter blocks that intersect Santa Monica boundary
        sm_blocks = gpd.overlay(blocks, santa_monica_boundary, how='intersection')
        
        print(f"Loaded {len(sm_blocks)} census blocks for Santa Monica")
        print(f"Unique tracts: {sorted(sm_blocks.TRACTCE20.unique())}")
        return sm_blocks
    
    def filter_relevant_tracts(self, blocks_gdf, santa_monica_boundary, tracts_gdf, min_overlap=0.3):
        """Filter to tracts with substantial Santa Monica coverage to prevent overcounting."""
        print(f"Filtering tracts with at least {min_overlap:.0%} overlap with Santa Monica...")
        
        # Debug: Check tract shapefile columns
        print(f"Available tract columns: {list(tracts_gdf.columns)}")
        
        # Find the correct tract ID column
        tract_id_col = None
        for col in ['TRACTCE20', 'TRACTCE', 'GEOID', 'GEOID20']:
            if col in tracts_gdf.columns:
                tract_id_col = col
                break
        
        if tract_id_col is None:
            raise ValueError(f"Cannot find tract ID column in {list(tracts_gdf.columns)}")
        
        print(f"Using tract ID column: {tract_id_col}")
        
        # Convert to projected CRS for accurate area calculations
        proj_crs = 'EPSG:3857'  # Web Mercator
        sm_boundary_proj = santa_monica_boundary.to_crs(proj_crs)
        tracts_proj = tracts_gdf.to_crs(proj_crs)
        
        tract_overlaps = []
        unique_tracts = blocks_gdf.TRACTCE20.unique()
        
        for tract_id in unique_tracts:
            try:
                # Get tract geometry - handle different possible tract ID formats
                tract_matches = tracts_proj[tracts_proj[tract_id_col].astype(str).str.contains(tract_id[-6:], na=False)]
                
                if len(tract_matches) == 0:
                    print(f"  Tract {tract_id}: Not found in tract shapefile")
                    continue
                    
                tract_geom = tract_matches.geometry.iloc[0]
                sm_geom = sm_boundary_proj.geometry.iloc[0]
                
                # Calculate overlap
                intersection = tract_geom.intersection(sm_geom)
                overlap_area = intersection.area if hasattr(intersection, 'area') else 0
                tract_area = tract_geom.area
                
                if tract_area > 0:
                    overlap_pct = overlap_area / tract_area
                else:
                    overlap_pct = 0
                
                tract_overlaps.append({
                    'TRACTCE20': tract_id,
                    'overlap_percentage': overlap_pct,
                    'overlap_area': overlap_area,
                    'tract_area': tract_area
                })
                
                print(f"  Tract {tract_id}: {overlap_pct:.1%} overlap")
                
            except Exception as e:
                print(f"  Error processing tract {tract_id}: {e}")
                continue
        
        # Filter to tracts with sufficient overlap
        relevant_tracts = [t['TRACTCE20'] for t in tract_overlaps if t['overlap_percentage'] >= min_overlap]
        excluded_tracts = [t['TRACTCE20'] for t in tract_overlaps if t['overlap_percentage'] < min_overlap]
        
        print(f"Using {len(relevant_tracts)} tracts with ≥{min_overlap:.0%} overlap")
        if excluded_tracts:
            print(f"Excluded {len(excluded_tracts)} tracts with low overlap: {excluded_tracts}")
        
        # If no tracts meet threshold, lower it and warn
        if len(relevant_tracts) == 0:
            print(f"WARNING: No tracts found with ≥{min_overlap:.0%} overlap. Lowering threshold to 10%...")
            relevant_tracts = [t['TRACTCE20'] for t in tract_overlaps if t['overlap_percentage'] >= 0.1]
            
            if len(relevant_tracts) == 0:
                print("WARNING: Still no tracts found. Using all tracts with any overlap...")
                relevant_tracts = [t['TRACTCE20'] for t in tract_overlaps if t['overlap_percentage'] > 0]
                
                if len(relevant_tracts) == 0:
                    print("ERROR: No tract overlaps found. Using all original tracts...")
                    relevant_tracts = list(unique_tracts)
        
        return relevant_tracts, tract_overlaps
    
    def get_census_demographics(self, relevant_tracts):
        """Fetch demographic data from Census API for filtered tracts only."""
        print(f"Fetching demographic data from Census API for {len(relevant_tracts)} relevant tracts...")
        
        all_tract_data = []
        all_pop_data = []
        
        for tract in relevant_tracts:
            try:
                # Get tract-level ACS data for income, housing, age
                acs_data = self.census.acs5.state_county_tract(
                    fields=('B19013_001E', 'B25077_001E', 'B01002_001E'),
                    state_fips=self.ca_fips,
                    county_fips='037',
                    tract=tract,
                    year=2022
                )
                
                if acs_data:
                    tract_demo = {
                        'TRACTCE20': tract,
                        'median_income': int(acs_data[0]['B19013_001E']) if acs_data[0]['B19013_001E'] else None,
                        'median_home_value': int(acs_data[0]['B25077_001E']) if acs_data[0]['B25077_001E'] else None,
                        'median_age': float(acs_data[0]['B01002_001E']) if acs_data[0]['B01002_001E'] else None
                    }
                    all_tract_data.append(tract_demo)
                
                # Get tract-level population data
                pop_data = self.census.pl.state_county_tract(
                    fields=('P1_001N', 'P1_003N', 'P1_004N', 'P1_005N', 'P1_006N'),
                    state_fips=self.ca_fips,
                    county_fips='037',
                    tract=tract,
                    year=2020
                )
                
                if pop_data:
                    tract_pop = {
                        'TRACTCE20': tract,
                        'total_pop': int(pop_data[0]['P1_001N'])
                    }
                    all_pop_data.append(tract_pop)
                    
            except Exception as e:
                print(f"Error fetching data for tract {tract}: {e}")
                continue
        
        tract_demographics = pd.DataFrame(all_tract_data)
        pop_demographics = pd.DataFrame(all_pop_data)
        
        print(f"Retrieved tract demographics: {len(tract_demographics)} tracts")
        print(f"Retrieved population data: {len(pop_demographics)} tracts")
        return tract_demographics, pop_demographics
    
    def distribute_population_geographically(self, blocks_gdf, pop_demographics, santa_monica_boundary, tract_overlaps):
        """Distribute tract population based on geographic area within Santa Monica."""
        print("Distributing tract population geographically by area...")
        
        # Convert to projected CRS for accurate area calculations
        proj_crs = 'EPSG:3857'
        blocks_proj = blocks_gdf.to_crs(proj_crs)
        sm_boundary_proj = santa_monica_boundary.to_crs(proj_crs)
        
        # Create overlap lookup
        overlap_dict = {t['TRACTCE20']: t['overlap_percentage'] for t in tract_overlaps}
        
        # Calculate area-weighted population for each block
        blocks_with_pop = blocks_proj.copy()
        
        for pop_col in ['total_pop']:
            blocks_with_pop[f'{pop_col}_allocated'] = 0.0
        
        for tract_id in pop_demographics['TRACTCE20'].unique():
            tract_data = pop_demographics[pop_demographics['TRACTCE20'] == tract_id].iloc[0]
            tract_blocks = blocks_with_pop[blocks_with_pop['TRACTCE20'] == tract_id]
            
            if len(tract_blocks) == 0:
                continue
            
            # Get overlap percentage for this tract
            overlap_pct = overlap_dict.get(tract_id, 1.0)
            
            # Calculate total area of Santa Monica portion within this tract
            total_sm_area_in_tract = 0
            block_areas = []
            
            for idx, block in tract_blocks.iterrows():
                try:
                    # Calculate intersection area with Santa Monica
                    intersection = block.geometry.intersection(sm_boundary_proj.geometry.iloc[0])
                    block_sm_area = intersection.area if hasattr(intersection, 'area') else 0
                    block_areas.append((idx, block_sm_area))
                    total_sm_area_in_tract += block_sm_area
                except Exception as e:
                    print(f"Error calculating area for block {idx}: {e}")
                    block_areas.append((idx, 0))
            
            # Distribute population proportionally by area, adjusted for tract overlap
            for pop_col in ['total_pop']:
                # Use only the portion of tract population that's actually in Santa Monica
                adjusted_tract_pop = tract_data[pop_col] * overlap_pct
                
                for idx, block_sm_area in block_areas:
                    if total_sm_area_in_tract > 0:
                        allocated_pop = adjusted_tract_pop * (block_sm_area / total_sm_area_in_tract)
                    else:
                        allocated_pop = 0
                    
                    blocks_with_pop.loc[idx, f'{pop_col}_allocated'] = allocated_pop
        
        # Convert back to original CRS
        blocks_with_pop = blocks_with_pop.to_crs(blocks_gdf.crs)
        
        # Verify total population
        total_allocated = blocks_with_pop['total_pop_allocated'].sum()
        print(f"Total allocated population: {total_allocated:,.0f}")
        
        return blocks_with_pop
    
    def assign_blocks_to_neighborhoods(self, blocks_gdf, neighborhoods_gdf):
        """Assign census blocks to neighborhood organizations using spatial join."""
        print("Assigning census blocks to neighborhoods...")
        
        # Convert to projected CRS for accurate spatial operations
        proj_crs = 'EPSG:3857'  # Web Mercator
        blocks_proj = blocks_gdf.to_crs(proj_crs)
        neighborhoods_proj = neighborhoods_gdf.to_crs(proj_crs)
        
        # Calculate block centroids in projected CRS
        blocks_proj['centroid'] = blocks_proj.geometry.centroid
        
        # Create centroids GeoDataFrame
        centroids_gdf = blocks_proj.copy()
        centroids_gdf.geometry = centroids_gdf.centroid
        
        # Spatial join using centroids
        joined = gpd.sjoin(centroids_gdf, neighborhoods_proj, how='left', predicate='within')
        
        assigned_count = joined['index_right'].notna().sum()
        print(f"Assigned {assigned_count} out of {len(joined)} blocks to neighborhoods")
        
        # Print assignment summary
        if 'neighbor' in joined.columns:
            assignment_summary = joined.groupby('neighbor', dropna=False).size()
            print("Blocks per neighborhood:")
            for neighborhood, count in assignment_summary.items():
                print(f"  {neighborhood}: {count} blocks")
        
        return joined
    
    def aggregate_demographics(self, joined_blocks, tract_demographics, neighborhoods_gdf):
        """Aggregate allocated demographic data by neighborhood."""
        print("Aggregating demographics by neighborhood...")
        
        # Merge tract-level data for income/housing
        blocks_with_demo = joined_blocks.merge(
            tract_demographics,
            on='TRACTCE20',
            how='left'
        )
        
        # Get neighborhood names mapping
        neighborhood_names = dict(zip(neighborhoods_gdf.index, neighborhoods_gdf['neighbor']))
        
        # Aggregate by neighborhood
        neighborhood_stats = []
        
        for idx in neighborhoods_gdf.index:
            neighborhood_name = neighborhood_names.get(idx, f"Neighborhood {idx}")
            mask = blocks_with_demo['index_right'] == idx
            neighborhood_blocks = blocks_with_demo[mask]
            
            if len(neighborhood_blocks) > 0:
                # Sum allocated population data
                total_pop = neighborhood_blocks['total_pop_allocated'].sum()
                
                if total_pop > 0:
                    stats = {
                        'neighborhood_id': idx,
                        'neighborhood_name': neighborhood_name,
                        'total_blocks': len(neighborhood_blocks),
                        'total_pop': int(total_pop),
                        'median_income': neighborhood_blocks['median_income'].median(),
                        'median_home_value': neighborhood_blocks['median_home_value'].median(),
                        'median_age': neighborhood_blocks['median_age'].median()
                    }
                    neighborhood_stats.append(stats)
                    print(f"  {neighborhood_name}: {int(total_pop):,} residents in {len(neighborhood_blocks)} blocks")
        
        if not neighborhood_stats:
            raise ValueError("No neighborhoods have population data. Check spatial assignments.")
        
        stats_df = pd.DataFrame(neighborhood_stats)
        
        return stats_df
    
    def calculate_malapportionment_metrics(self, stats_df):
        """Calculate various malapportionment metrics."""
        print("\nCalculating malapportionment metrics...")
        
        if len(stats_df) < 2:
            print("Warning: Only one neighborhood has data. Cannot calculate meaningful metrics.")
            return {}, stats_df
        
        # Coefficient of Variation
        mean_pop = stats_df['total_pop'].mean()
        std_pop = stats_df['total_pop'].std()
        cv = (std_pop / mean_pop) * 100 if mean_pop > 0 else 0
        
        # Tyler's Extremity Ratio
        max_pop = stats_df['total_pop'].max()
        min_pop = stats_df['total_pop'].min()
        extremity_ratio = max_pop / min_pop if min_pop > 0 else float('inf')
        
        # Relative Deviations
        ideal_pop = stats_df['total_pop'].sum() / len(stats_df)
        stats_df['relative_deviation'] = ((stats_df['total_pop'] - ideal_pop) / ideal_pop) * 100
        stats_df['representation_ratio'] = ideal_pop / stats_df['total_pop']
        
        # Gini Coefficient
        sorted_pops = sorted(stats_df['total_pop'])
        n = len(sorted_pops)
        if n > 1:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_pops)) / (n * np.sum(sorted_pops)) - (n + 1) / n
        else:
            gini = 0
        
        metrics = {
            'coefficient_of_variation': cv,
            'extremity_ratio': extremity_ratio,
            'max_relative_deviation': stats_df['relative_deviation'].abs().max(),
            'gini_coefficient': gini,
            'mean_population': mean_pop,
            'std_population': std_pop,
            'ideal_population': ideal_pop
        }
        
        return metrics, stats_df
    
    def create_visualizations(self, stats_df, metrics, neighborhoods_gdf):
        """Create visualizations for the analysis."""
        print("\nCreating visualizations...")
        
        if len(stats_df) == 0:
            print("No data to visualize")
            return
        
        # Set up plotting style  
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Santa Monica Neighborhood Organization Malapportionment Analysis (Fixed)', fontsize=16)
        
        # 1. Population by Neighborhood
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(stats_df)), stats_df['total_pop'], color='steelblue')
        ax1.axhline(y=metrics.get('ideal_population', 0), color='red', linestyle='--', label='Ideal Population')
        ax1.set_xlabel('Neighborhood')
        ax1.set_ylabel('Population')
        ax1.set_title('Population by Neighborhood Organization')
        ax1.set_xticks(range(len(stats_df)))
        ax1.set_xticklabels(stats_df['neighborhood_name'], rotation=45, ha='right')
        ax1.legend()
        
        # 2. Relative Deviation
        ax2 = axes[0, 1]
        colors = ['red' if x < 0 else 'green' for x in stats_df['relative_deviation']]
        ax2.bar(range(len(stats_df)), stats_df['relative_deviation'], color=colors)
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='±5% threshold')
        ax2.axhline(y=-5, color='orange', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Neighborhood')
        ax2.set_ylabel('Relative Deviation (%)')
        ax2.set_title('Deviation from Ideal Population')
        ax2.set_xticks(range(len(stats_df)))
        ax2.set_xticklabels(stats_df['neighborhood_name'], rotation=45, ha='right')
        ax2.legend()
        
        # 3. Income Distribution
        ax3 = axes[0, 2]
        valid_income = stats_df['median_income'].dropna()
        if len(valid_income) > 0:
            ax3.bar(range(len(stats_df)), stats_df['median_income'], color='green')
            ax3.set_xlabel('Neighborhood')
            ax3.set_ylabel('Median Income ($)')
            ax3.set_title('Median Household Income by Neighborhood')
            ax3.set_xticks(range(len(stats_df)))
            ax3.set_xticklabels(stats_df['neighborhood_name'], rotation=45, ha='right')
        
        # 4. Home Values
        ax4 = axes[1, 0]
        valid_values = stats_df['median_home_value'].dropna()
        if len(valid_values) > 0:
            ax4.bar(range(len(stats_df)), stats_df['median_home_value'], color='purple')
            ax4.set_xlabel('Neighborhood')
            ax4.set_ylabel('Median Home Value ($)')
            ax4.set_title('Median Home Value by Neighborhood')
            ax4.set_xticks(range(len(stats_df)))
            ax4.set_xticklabels(stats_df['neighborhood_name'], rotation=45, ha='right')
        
        # 5. Metrics Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        metrics_text = f"""Malapportionment Metrics:

Coefficient of Variation: {metrics.get('coefficient_of_variation', 0):.2f}%
(Below 10% is generally acceptable)

Extremity Ratio: {metrics.get('extremity_ratio', 0):.2f}:1
(Above 2:1 indicates substantial inequality)

Max Relative Deviation: {metrics.get('max_relative_deviation', 0):.2f}%
(Legal standard: ±5% from ideal)

Gini Coefficient: {metrics.get('gini_coefficient', 0):.3f}
(0 = perfect equality, 1 = maximum inequality)

Mean Population: {metrics.get('mean_population', 0):,.0f}
Ideal Population: {metrics.get('ideal_population', 0):,.0f}"""
        
        ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('santa_monica_malapportionment_analysis_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_report(self, stats_df, metrics):
        """Generate a summary report."""
        print("\n" + "="*60)
        print("SANTA MONICA NEIGHBORHOOD MALAPPORTIONMENT ANALYSIS REPORT (FIXED)")
        print("="*60)
        
        if len(stats_df) == 0:
            print("No data available for analysis.")
            return stats_df
        
        print("\nPOPULATION DISTRIBUTION:")
        total_pop = 0
        for _, row in stats_df.iterrows():
            total_pop += row['total_pop']
            print(f"{row['neighborhood_name']}: {row['total_pop']:,} residents "
                  f"({row['relative_deviation']:+.1f}% from ideal)")
        
        print(f"TOTAL ALLOCATED: {total_pop:,}")
        print(f"Expected Santa Monica total: 93,076")
        print(f"Coverage: {total_pop/93076:.1%}")
        
        print(f"\nKEY FINDINGS:")
        print(f"- Coefficient of Variation: {metrics.get('coefficient_of_variation', 0):.1f}%")
        
        cv = metrics.get('coefficient_of_variation', 0)
        if cv > 10:
            print("  ⚠️  EXCEEDS acceptable threshold of 10%")
        else:
            print("  ✓ Within acceptable range")
            
        print(f"- Population Extremity Ratio: {metrics.get('extremity_ratio', 0):.2f}:1")
        
        ratio = metrics.get('extremity_ratio', 0)
        if ratio > 2:
            print("  ⚠️  SUBSTANTIAL inequality detected")
        else:
            print("  ✓ Within reasonable range")
            
        print(f"- Maximum Deviation: {metrics.get('max_relative_deviation', 0):.1f}%")
        
        max_dev = metrics.get('max_relative_deviation', 0)
        if max_dev > 5:
            print("  ⚠️  EXCEEDS legal standard of ±5%")
        else:
            print("  ✓ Meets legal standard")
        
        # Save detailed results
        stats_df.to_csv('neighborhood_demographics_fixed.csv', index=False)
        print("\nDetailed results saved to 'neighborhood_demographics_fixed.csv'")
        
        return stats_df

def main():
    """Main analysis execution with geographic corrections."""
    CENSUS_API_KEY = os.environ.get('CENSUS_API_KEY')
    if not CENSUS_API_KEY:
        raise ValueError("CENSUS_API_KEY environment variable is required")
    analysis = SantaMonicaNeighborhoodAnalysis(CENSUS_API_KEY)
    
    try:
        # Load neighborhood boundaries
        neighborhoods = analysis.download_neighborhood_boundaries()
        
        # Get Santa Monica city boundary for proper filtering
        santa_monica_boundary = analysis.download_santa_monica_boundary()
        
        # Download census tracts for geographic analysis
        tracts = analysis.download_census_tracts()
        
        # Download census blocks within Santa Monica
        blocks = analysis.download_census_blocks(santa_monica_boundary)
        
        # Filter to relevant tracts (prevents overcounting)
        relevant_tracts, tract_overlaps = analysis.filter_relevant_tracts(
            blocks, santa_monica_boundary, tracts, min_overlap=0.3
        )
        
        # Get demographic data for relevant tracts only
        tract_demographics, pop_demographics = analysis.get_census_demographics(relevant_tracts)
        
        # Distribute population geographically by area
        blocks_with_allocated_pop = analysis.distribute_population_geographically(
            blocks, pop_demographics, santa_monica_boundary, tract_overlaps
        )
        
        # Assign blocks to neighborhoods
        joined_blocks = analysis.assign_blocks_to_neighborhoods(blocks_with_allocated_pop, neighborhoods)
        
        # Aggregate demographics by neighborhood
        stats_df = analysis.aggregate_demographics(
            joined_blocks, tract_demographics, neighborhoods
        )
        
        # Calculate malapportionment metrics
        metrics, stats_df_with_metrics = analysis.calculate_malapportionment_metrics(stats_df)
        
        # Generate visualizations and report
        analysis.create_visualizations(stats_df_with_metrics, metrics, neighborhoods)
        analysis.generate_report(stats_df_with_metrics, metrics)
        
        print(f"\nAnalysis complete! Found data for {len(stats_df)} neighborhoods.")
        print("Geographic corrections applied to prevent population overcounting.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()