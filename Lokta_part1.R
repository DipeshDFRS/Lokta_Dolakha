
# Load required libraries ####
library(raster)            # Read raster data
library(sf)                # Spatial vector data handling
library(e1071)             # Misc ML utilities
library(ExtractTrainData)  # Extract training data
library(readxl)            # Read Excel files
library(dplyr)             # Data manipulation
library(sp)                # Spatial objects


# Set coordinate projections ####
wgs1984.proj <- st_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
UTMZ45 <- st_crs("+proj=utm +zone=45 +datum=WGS84 +units=m +no_defs")

#Set working directory####
setwd("D:/dolakha")

#Load study area boundary####
dkh_boundary<-st_read("predictors/Dolkha_buffer_100m.shp")
dkh_boundary_crs <- st_transform(dkh_boundary, crs = wgs1984.proj )

##Laod DEM data####
dem<-raster("predictors/srtm_dolakha.tif")
slope<-terrain(dem, opt= "slope", unit= "degrees", neighbors=8)
aspect <-terrain(dem, opt= "aspect",unit= "degrees", neighbors=8)

##Load worldclim data####
worldclim<-raster('predictors/worldclim_lokta.tif')
worldclim_crop<-crop(worldclim,dkh_boundary_crs)
worldclim_re<-resample(worldclim_crop,dem,method ="bilinear")

#Load raster layers####

##NDVI####
ndvi <- raster('predictors/ndvi_lokta.tif')
ndvi_crop<-crop(ndvi,dkh_boundary_crs)
ndvi_re <- resample(ndvi_crop, dem, method = "bilinear")

##MSAVI2####
msavi2<- raster('predictors/msavi2_lokta.tif')
msavi2_crop<-crop(msavi2,dkh_boundary_crs)
msavi2_re <- resample(msavi2_crop, dem, method = "bilinear")

##Silt####
Silt<- raster('predictors/silt515_lokta.tif')
Silt_crop<-crop(Silt,dkh_boundary_crs)
Silt_re <- resample(Silt_crop, dem, method = "bilinear")

##clay####
clay<- raster('predictors/clay515_lokta.tif')
clay_crop<-crop(clay,dkh_boundary_crs)
clay_re <- resample(clay_crop, dem, method = "bilinear")

##sand####
sand<- raster('predictors/sand515_lokta.tif')
sand_crop<-crop(sand,dkh_boundary_crs)
sand_re <- resample(sand_crop, dem, method = "bilinear")

##ph####
ph<- raster('predictors/ph515_lokta.tif')
ph_crop<-crop(ph,dkh_boundary_crs)
ph_re <- resample(ph_crop, dem, method = "bilinear")

##nitrogen####
nitrogen <- raster('predictors/nitrogen515_lokta.tif')
nitrogen_crop<-crop(nitrogen,dkh_boundary_crs)
nitrogen_re <- resample(nitrogen_crop, dem, method = "bilinear")

##Carbon####
Carbon <- raster('predictors/SOC515_lokta.tif')
Carbon_crop<-crop(Carbon,dkh_boundary_crs)
Carbon_re <- resample(Carbon_crop, dem, method = "bilinear")


#Load species presence/absence data ####

pa_points_1 <-  read_excel(file.choose(), 1) ## Load Lokta_GPS points.xlsx ####
pa_points_2 <-  read_excel(file.choose(), 2) ## Load Lokta_plot detail.xlsx ####
pa_points_3 <-  read_excel(file.choose(), 3) ## Load Lokta_plot detail.xlsx ####

#Clean and rename columns####
pa_points_1 <- pa_points_1 [-93, - c(1,4,5,6)]%>% 
  rename(lon = `GPS X`,lat = `GPS Y`,p_a = `Lokta (Y/N)`) %>% 
  mutate(p_a = ifelse(p_a == "Yes", "1", "0"))

pa_points_2 <- pa_points_2 [, - 1]%>% 
  rename(lon = `X`,lat = `Y`,p_a = `Presence of lokta`) %>% 
  mutate(p_a = ifelse(p_a == "Yes", "1", "0"))

pa_points_3 <- pa_points_3 [, - c(1,4,5)]%>% 
  rename(lon =  "Latitude" ,lat = "Longitude",p_a = `Presence of lokta`) %>% 
  mutate( p_a = ifelse(p_a == "Yes", "1", "0"))


##Merge all presence/absence data ####
pa_points <- bind_rows(pa_points_1, pa_points_2, pa_points_3)
write.csv(pa_points,"Data/Lokta_PA.csv")
pa_points1 <- read.csv("Data/Lokta_PA.csv")


## Convert to spatial ####
pa_points_shp <- st_as_sf(data.frame(pa_points1), coords = c("lon", "lat"))
st_crs(pa_points_shp) <-  UTMZ45
pa_points_shp <- st_transform(pa_points_shp, wgs1984.proj)


# Create raster stack ####
predictors <-stack(worldclim_re,dem,slope,aspect,ndvi_re, 
                   msavi2_re,Silt_re, sand_re, clay_re, ph_re,
                   nitrogen_re, Carbon_re)


# Convert raster stack to data frame####

dt = as.data.frame(predictors, xy = T, na.rm = T, header = T) %>% 
  rename(elevation = srtm_dolakha1)


db1 <- dt %>%
  rename(
    BIO1 = layer.1,BIO2 = layer.2,BIO3 = layer.3, BIO4 = layer.4,
    BIO5 = layer.5,BIO6 = layer.6,BIO7 = layer.7, BIO8 = layer.8,
    BIO9 = layer.9,BIO10 = layer.10,BIO11 = layer.11, BIO12 = layer.12,
    BIO13 = layer.13, BIO14 = layer.14, BIO15 = layer.15,
    BIO16 = layer.16, BIO17 = layer.17, BIO18 = layer.18, BIO19 = layer.19,
    MSAVI2=msavi2_lokta, Silt=slit515_lokta, Sand=sand515_lokta, 
    Clay= clay515_lokta, pH= ph515_lokta,
    Nitrogen =nitrogen515_lokta, Carbon = Carbon515_lokta)

#Extract raster values at PA points
e_points = raster::extract(predictors,pa_points_shp)

#Combine the extracted values with the original data frame
pa_points <- cbind(pa_points_shp, e_points)

#Convert to dataframe
pa_points_df = as.data.frame(pa_points, xy = T, na.rm = T)

#Export####
write.csv(db1,"Data/lokta_all_coords.csv")
write.csv(pa_points_df,"Data/lokta.csv")
st_write(db1,"Data/lokta_all_coords.shp")


## Import the nearest distance calculated in ArcGis Pro 

# Specify the path to the GDB and the layer to extract
gdb_path <- "C:/Documents/ArcGIS/Projects/lokta_distance/lokta_distance.gdb"
layer_name <- "lokta_all_Proj_ExportFeature"  # Replace with the layer name

# Read the data from the GDB
gdb_data <- st_read(dsn = gdb_path, layer = layer_name)
gdb_data_transformed <- st_transform(gdb_data, crs = 4326)

# Write the transformed data to a shapefile
st_write(gdb_data_transformed, "Data/Lokta_all_pred.shp")

#End of Part 1####
