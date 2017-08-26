# Path planning write up
## Quick overview
The main challenges involved in this project involved generating paths
that did not violate any of the comfort parameters - jerk, acceleration, collisions, etc.
Therefore given that the car consumed a path coordinate at a time-step of 0.2 seconds it was important to
use the spline package to interpolate at a regularity of 30m. 

This was achieved by the code snippet below. These waypoints were then intersplined to smoothen the ego vechicles trajectory. Line 377-399 in 
main.cpp. 
```
...
vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
...
``` 

To avoid collisions, each time the ego vechicle was behind a car in its lane less than 30m away - within the range of the sensor, it would slow 
down the car at a rate within the jerk comfort parameters.  This detection was achieved by the code snippet below, in lines 303-319.
```
...
//car in my lane
float d = sensor_fusion[i][6];
if (d < (2 + 4 * lane + 2) && d > (2 + 4 * lane - 2))
{
	double vx = sensor_fusion[i][3];
	double vy = sensor_fusion[i][4];
	double check_speed = sqrt(vx * vx + vy * vy);
	double check_car_s = sensor_fusion[i][5];
	check_car_s += ((double)prev_size * 0.02 * check_speed);
	if ((check_car_s > car_s) && ((check_car_s - car_s) < 30))
	{ 
...
```

Once this slow down manuovre was initiated, the ego car would look to other lanes. Should the lane be empty the vechicle would 
set the target lane to this value. However should the lane have a vehicle within it, ahead of the ego vehicle moving faster than the current
car currently ahead of it, then switch over to this lane. This is shown in the code snippet below, in line 109-206 of main.cpp


```
...
//find max car distance
if(!lane_change_heurestic.empty()){
	auto max_dist = max_element(lane_change_heurestic.begin(), lane_change_heurestic.end());
	int temp_lane = distance(lane_change_heurestic.begin(), max_dist);
	if(temp_lane<=2 && temp_lane>=0){
		lane_to_go_to = temp_lane;
	}
}
...
```

