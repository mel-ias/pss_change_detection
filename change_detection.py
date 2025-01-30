import argparse
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rasterio import features
from rasterio.mask import mask
from shapely.geometry import LineString
from mpl_toolkits import axes_grid1

# Function to load a raster including NoData value
def load_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Load only the first band (elevation values)
        nodata = src.nodata  # Query NoData value
        transform = src.transform
        crs = src.crs
        profile = src.profile
    return data, transform, crs, profile, nodata

# Function to resample a raster to a target raster
def resample_raster(source, source_transform, source_crs, target_shape, target_transform, target_crs, nodata_value):
    destination = np.empty(target_shape, dtype=source.dtype)
    reproject(
        source,
        destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )
    destination[destination == nodata_value] = np.nan
    return destination

# Function to apply mask (shapefile) to the raster
def apply_mask(raster_data, transform, shapefile, nodata_value):
    shapes = [geom for geom in shapefile.geometry]
    out_image, out_transform = mask(raster_data, shapes, crop=True, nodata=nodata_value)
    out_image[out_image == nodata_value] = np.nan
    return out_image[0], out_transform

# IQR 1.5 Filter to remove outliers
# def removeOutliers(arr, outlierConstant=1.5):
#     lower_quartile = np.nanpercentile(arr, 25)
#     upper_quartile = np.nanpercentile(arr, 75)
#     IQR = (upper_quartile - lower_quartile) * outlierConstant
#     lower_bound = lower_quartile - IQR
#     upper_bound = upper_quartile + IQR
#     filtered_data = arr.copy()
#     filter_mask = (arr < lower_bound) | (arr > upper_bound)
#     filtered_data[filter_mask] = np.nan

#     num_original_valid_values = np.sum(~np.isnan(arr))
#     num_filtered_values = np.sum(filter_mask & ~np.isnan(arr))
#     percent_filtered = (num_filtered_values / num_original_valid_values) * 100
#     print("Remove coarse outliers from DoD using IQR filtering with constant (k):", outlierConstant, "percentage filtered:", percent_filtered)
#     return filtered_data, percent_filtered

# Create hillshade for the independent DEM
def hillshade(array, azimuth=315, angle_altitude=45):
    x, y = np.gradient(array)
    slope = np.pi / 2. - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuth_rad = azimuth * np.pi / 180.
    altitude_rad = angle_altitude * np.pi / 180.
    shaded = np.sin(altitude_rad) * np.sin(slope) + np.cos(altitude_rad) * np.cos(slope) * np.cos(azimuth_rad - aspect)
    return 255 * (shaded + 1) / 2

# Function to extract elevation along a polyline
def extract_profile_from_dem(dem_path, polyline, num_points=100):
    with rasterio.open(dem_path) as dem:
        line = polyline.geometry.iloc[0]  # Take the first row in geometry
        points = [line.interpolate(i / num_points, normalized=True) for i in range(num_points + 1)]
        coords = [(point.x, point.y) for point in points]
        heights = [val[0] for val in dem.sample(coords)]  # Extract values from DEM
        distances = [line.project(point) for point in points]  # Distances along the line
    return distances, heights

# Function to add a vertical color bar to an image plot
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# Main function to process DEM and Shapefile paths
def main():
    
    parser = argparse.ArgumentParser(description='Process DEM and Shapefile paths.')
    parser.add_argument('--hillshade_dem_path', type=str, required=True, help='Path to the hillshade DEM file')
    parser.add_argument('--mask_shapefile_path', type=str, required=True, help='Path to the mask shapefile that outlines the area of comparison')
    parser.add_argument('--output_path_print', type=str, required=True, help='Output path for results')
    parser.add_argument('--prefix_dem1', type=str, required=True, help='First year identifier to be added to the plot caption')
    parser.add_argument('--prefix_dem2', type=str, required=True, help='Second year identifier to be added to the plot caption')
    parser.add_argument('--profile_shapefile_path', type=str, default="", help='In case one will investigate the elevation change along a profile, provide the path to the profile as shapefile (polyline)')
    parser.add_argument('--dem1_path', type=str, required=True, help='Path to the first DEM file')
    parser.add_argument('--dem2_path', type=str, required=True, help='Path to the second DEM file')
    parser.add_argument('--outline_shape', type=str, required=True, help='In case on will overlay the graphic with outlines, e.g. from a glacier, provide the path to the outline shapefile')
    args = parser.parse_args()

    # Load DEMs and shapefile
    hillshade_dem_path = args.hillshade_dem_path
    mask_shapefile_path = args.mask_shapefile_path
    output_path_print = args.output_path_print
    prefix_dem1 = args.prefix_dem1
    prefix_dem2 = args.prefix_dem2
    profile_shapefile_path = args.profile_shapefile_path
    dem1_path = args.dem1_path
    dem2_path = args.dem2_path
    outline_shape = args.outline_shape

    # Ensure common extent for DEMs
    with rasterio.open(dem2_path) as src, rasterio.open(dem1_path) as src_to_crop:
        src_affine = src.meta.get("transform")
        band = src.read(1)
        band[np.where(band != src.nodata)] = 1
        geoms = [geometry for geometry, raster_value in features.shapes(band, transform=src_affine) if raster_value == 1]
        out_img, out_transform = mask(dataset=src_to_crop, shapes=geoms, crop=True)
        with rasterio.open(dem1_path + "_clipped_dem2.tif", 'w', driver='GTiff', height=out_img.shape[1], width=out_img.shape[2], count=src.count, crs=src.crs, dtype=out_img.dtype, transform=out_transform) as dst:
            dst.write(out_img)
    dem1_path = dem1_path + "_clipped_dem2.tif"

    if profile_shapefile_path:
        polyline = gpd.read_file(profile_shapefile_path)
        distances_dem1, heights_dem1 = extract_profile_from_dem(dem1_path, polyline)
        distances_dem2, heights_dem2 = extract_profile_from_dem(dem2_path, polyline)
        plt.figure(figsize=(5, 5))
        plt.plot(distances_dem1, heights_dem1, label=prefix_dem1, color='blue')
        plt.plot(distances_dem2, heights_dem2, label=prefix_dem2, color='red')
        plt.fill_between(distances_dem1, heights_dem1, heights_dem2, color='gray', alpha=0.3, label='diff')
        plt.xlabel('Elevation Profile (m)')
        plt.ylabel('Elevation (m)')
        plt.title('Elevation Profile Deviation ' + prefix_dem1 + '/' + prefix_dem2)
        plt.legend()
        plt.grid(True)
        if output_path_print is not None:
            plt.savefig(output_path_print + "_elevation_profile_" + prefix_dem1 + "_" + prefix_dem2 + ".png", dpi=300, bbox_inches='tight')
        plt.close()

    # Load DEMs and shapefile
    dem1, dem1_transform, dem1_crs, dem1_profile, dem1_nodata = load_raster(dem1_path)
    dem2, dem2_transform, dem2_crs, dem2_profile, dem2_nodata = load_raster(dem2_path)
    mask_shapefile = gpd.read_file(mask_shapefile_path)

    # Replace NoData values with NaN
    if dem1_nodata is not None:
        dem1[dem1 == dem1_nodata] = np.nan
    if dem2_nodata is not None:
        dem2[dem2 == dem2_nodata] = np.nan

    # Apply mask to the raster
    with rasterio.open(dem1_path) as src1:
        masked_dem1, masked_dem1_transform = apply_mask(src1, dem1_transform, mask_shapefile, dem1_nodata)

    # Resample DEM2 to the resolution of DEM1
    dem2_resampled = resample_raster(dem2, dem2_transform, dem2_crs, masked_dem1.shape, masked_dem1_transform, dem1_crs, dem2_nodata)

    # Calculate the difference
    dem_diff = dem2_resampled - masked_dem1
    # dem_diff, percent_filtered = removeOutliers(dem_diff, outlierConstant=50) # the higher the constant, the less values are filtered 

    # Load DEM for hillshade calculation
    hillshade_dem, hillshade_transform, hillshade_crs, hillshade_profile, hillshade_nodata = load_raster(hillshade_dem_path)

    # Calculate hillshade for the independent DEM
    hillshade_array = hillshade(hillshade_dem)

    # Reproject polyline if CRS does not match
    if profile_shapefile_path and polyline.crs != masked_dem1_transform:
        polyline = polyline.to_crs(dem1_crs)

    if outline_shape:
        glacier_outline_loaded = gpd.read_file(outline_shape)
        if glacier_outline_loaded.crs != masked_dem1_transform:
            glacier_outline_loaded = glacier_outline_loaded.to_crs(dem1_crs)

    # Mask areas outside dem_diff with NaN
    dem_diff_masked = np.where(np.isnan(dem_diff), np.nan, dem_diff)

    # Visualization: Difference and Hillshade
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.1)
    ax = fig.add_subplot(gs[0])
    ax.imshow(hillshade_array, cmap='gray', alpha=0.5, extent=(
        hillshade_transform[2],
        hillshade_transform[2] + hillshade_transform[0] * hillshade_dem.shape[1],
        hillshade_transform[5] + hillshade_transform[4] * hillshade_dem.shape[0],
        hillshade_transform[5]
    ))
    diff_img = ax.imshow(dem_diff_masked, cmap='RdYlBu_r', alpha=0.8, extent=(
        masked_dem1_transform[2],
        masked_dem1_transform[2] + masked_dem1_transform[0] * masked_dem1.shape[1],
        masked_dem1_transform[5] + masked_dem1_transform[4] * masked_dem1.shape[0],
        masked_dem1_transform[5]
    ))

    # Add polyline
    if profile_shapefile_path:
        for _, row in polyline.iterrows():
            x, y = row.geometry.xy
            plt.plot(x, y, color='black', linewidth=1, linestyle='dashed', label='Polyline')

    # Add glacier outline
    if outline_shape:
        for _, row in glacier_outline_loaded.iterrows():
            if row.geometry.geom_type == 'Polygon':
                x, y = row.geometry.exterior.xy
                plt.plot(x, y, color='blue', linewidth=1)
            elif row.geometry.geom_type == 'MultiPolygon':
                for polygon in row.geometry:
                    x, y = polygon.exterior.xy
                    plt.plot(x, y, color='blue', linewidth=1)
            else:
                print(f"Geometry type {row.geometry.geom_type} is not supported.")

    ax.set_title('DoD ' + prefix_dem1 + '/' + prefix_dem2)
    cbar = add_colorbar(diff_img)
    cbar.set_label('Elevation Change (m)', rotation=90, labelpad=15)
    diff_img.set_clim(-50, 50)
    ax.set_xlim(masked_dem1_transform[2], masked_dem1_transform[2] + masked_dem1_transform[0] * masked_dem1.shape[1])
    ax.set_ylim(masked_dem1_transform[5] + masked_dem1_transform[4] * masked_dem1.shape[0], masked_dem1_transform[5])

    if output_path_print is not None:
        plt.savefig(output_path_print + "_dod_" + prefix_dem1 + "_" + prefix_dem2 + ".png", dpi=300, bbox_inches='tight')
    plt.close()

    # Histogram of changes
    plt.figure(figsize=(5, 5))
    valid_diff = dem_diff_masked[~np.isnan(dem_diff_masked)]
    plt.hist(valid_diff, bins=50, color='blue', alpha=0.8, edgecolor='w')
    mean_diff = np.nanmean(valid_diff)
    median_diff = np.nanmedian(valid_diff)
    min_diff = np.nanmin(valid_diff)
    max_diff = np.nanmax(valid_diff)
    plt.axvline(mean_diff, color='red', linestyle='dashed', linewidth=1, label=f'mean: {mean_diff:.2f}')
    plt.axvline(median_diff, color='orange', linestyle='dashed', linewidth=1, label=f'median: {median_diff:.2f}')
    plt.axvline(min_diff, color='green', linestyle='dashed', linewidth=1, label=f'min: {min_diff:.2f}')
    plt.axvline(max_diff, color='purple', linestyle='dashed', linewidth=1, label=f'max: {max_diff:.2f}')
    plt.legend()
    plt.title('Histogram of Elevation Changes')
    plt.xlabel('Elevation Change (m)')
    plt.ylabel('Number of Pixels')
    plt.grid()
    if output_path_print is not None:
        plt.savefig(output_path_print + "_histogram_elev_changes_" + prefix_dem1 + "_" + prefix_dem2 + ".png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
