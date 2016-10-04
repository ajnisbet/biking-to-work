# Biking to Work

The code behind my [blog post](https://www.andrewnisbet.nz/blog/bike-radius). 

Requires a Project OSRM HTTP instance running, and some OpenStreetMap data. I used `California.osm.pbf` from [Geofabrik](http://download.geofabrik.de/north-america.html), and docker for OSRM:
```bash
docker run 
	-p 5000:5000
	--name osrm-api
	-v ~/biking-to-work/profile.lua:/osrm-build/profile.lua
	-v ~/biking-to-work/osrm-data/:/osrm-data/
	cartography/osrm-backend-docker:latest
	osrm
	California
