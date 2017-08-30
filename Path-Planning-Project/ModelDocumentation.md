# Path planning write up
## Quick overview
The main challenges involved in this project involved generating paths
that did not violate any of the comfort parameters - jerk, acceleration, collisions, etc.
Therefore given that the car consumed a path coordinate at a time-step of 0.2 seconds it was important to
use the spline package to interpolate at a regularity of 30m. This had to be achieved whilst making use of a simple finite state machine
that guided the car throught the cluttered highway. One safe lane change at a time.

Jerk violation and comfort was achieved in part by the code segment below. 
Waypoints 30m were layed out and then intersplined to smoothen the ego vechicles trajectory as shown in line 377-399 in 
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

Once this slow down manuovre was initiated, the ego car would look to other lanes using a simple Finite State Machine.
Using the sensor fusion data, it would find the vehicles closest to it in adjacent lanes. This list of adjacent cars within the range of 30-300m (values picked randomly) would then be used to perform the most effecient lane change manouvre. 

Each of this cars is then evaluated using a simple heuristic with the simple logic encoded in the code snippet below:
```
...
check_car_s += ((double)prev_size * 0.02 * check_speed);
car_s +=  ((double)prev_size * 0.02 * current_speed_of_ego);

if((check_car_s < car_s) && abs(car_s-check_car_s) > 20 && (check_speed < current_speed_of_ego))
{
	lane_change_heurestic[i]+=15;
}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s > 25)){
	lane_change_heurestic[i]+=30;
}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s > 70)){
	lane_change_heurestic[i]+=60;
}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s > 105)){
	lane_change_heurestic[i]+=120;
}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s > 140)){
	lane_change_heurestic[i]+=1e3;
}else{
	lane_change_heurestic[i] -=10;
}
}
//if lane is empty, good
if(cars_in_lane == 0){
lane_change_heurestic[i] = 1e4;
}
...
````

* The lane with the highest heuristic score is then returned by the function.

 This is shown in the code snippet below, in line 109-206 of main.cpp


```
...
vector<int> final_lanes;
	for(pair<double, int> element : lane_sorted){
		final_lanes.push_back(element.second);
	}
...
```
Double lane changes are not allowed by the ego car logic and the next best lane heuristic is picked as shown in the code snippet below:
```
...
	if(abs(lane_to_go_to-current_lane)>=2){
		lane_to_go_to = final_lanes[1];
	}
	
	return lane_to_go_to;
...
```
## Future work
Given the time constraints other lane changing logic could have been coded into the vehicle to optimize
its speed around the track.

Other methods include using a neural network for behavioural cloning given the sensor fusion data and training data and using a deep Q-learner to optimize model parameters, again using the sensor fusion data as areward signal. The reward signal being the average speed of the vehicle per km of the track.
