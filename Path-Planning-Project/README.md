# Path planning write up
## Quick overview
The main challenge within this project involved generating safe paths
that did not violate any of the self driving vehicles comfort parameters which included - jerk, acceleration, collisions, etc. 
This had to be achieved whilst making use of a simple finite state machine
that guided the car throught the cluttered highway - one safe lane change at a time.

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
for(int i = 0; i < closest_vehicles.size(); i++){
	int cars_in_lane = closest_vehicles[i].size();
	for(int j =0; j < cars_in_lane; j++){
		double vx = closest_vehicles[i][j][3];
		double vy = closest_vehicles[i][j][4];
		double check_speed = sqrt(vx * vx + vy * vy);
		double check_car_s = closest_vehicles[i][j][5];
		check_car_s += ((double)prev_size * 0.02 * check_speed);
		car_s +=  ((double)prev_size * 0.02 * current_speed_of_ego);
		if((check_car_s < car_s) && abs(car_s-check_car_s) > 20 && (check_speed < current_speed_of_ego))
			{
				lane_change_heurestic[i]+=20;
			}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s) < 35){
				lane_change_heurestic[i]-=-60;
			}else if((check_car_s>car_s) && (check_speed > current_speed_of_ego) && (check_car_s-car_s) > 35){
				lane_change_heurestic[i]+=60;
			}
		}
	}
	//if lane is empty, good
	if(cars_in_lane == 0){
		lane_change_heurestic[i] += 1e4;
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

Other methods that could have been used to facilitate effective lane changes; include using a neural network for behavioural cloning. The sensor fusion data would be the input signal used for training, the cost function would be
the comfort parameter signal. A second more advanced method would involve the use of a deep Q-learner to optimize model parameters, again 
using the comfort parameters as a reward signal alongside the average speed around the track. The state would be cars current sensor fusion data, the actions space would include lane change.
