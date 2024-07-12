# SPDX-License-Identifier: BSD-3-Clause

# flake8: noqa F401
from collections.abc import Callable


import numpy as np

from vendeeglobe import (
    Checkpoint,
    Heading,
    Instructions,
    Location,
    Vector,
    config,
)
from vendeeglobe.utils import distance_on_surface, goto

class Bot:
    """
    This is the ship-controlling bot that will be instantiated for the competition.
    """

    def __init__(self):
        self.team = "SLaB"  # This is your team name
        # This is the course that the ship has to follow
        self.course = [
            Checkpoint(latitude=43.797109, longitude=-11.264905, radius=100),
            Checkpoint(latitude=42, longitude=-29.0518, radius=100),
            Checkpoint(latitude=20.256440868994147,longitude=-62.734644270082256, radius=60),
            Checkpoint(latitude=15.265393163063342,longitude=-65.72726404185893, radius=40),
            Checkpoint(latitude=10.426543935069509,longitude=-80.92130587774096, radius=40),
            Checkpoint(latitude=6.671194522393581,longitude=-78.33150587841799, radius=40),
            Checkpoint(latitude=1.5841160481860714,longitude=-91.73960230587916, radius=100.0), # over glapagos islands
            Checkpoint(latitude=2.806318, longitude=-168.943864, radius=1990.0),
            Checkpoint(latitude=-32.564847188268974, longitude=172.2733733906634, radius=100.0), # over new seeland
            Checkpoint(latitude=-45.052286, longitude=146.214572, radius=100.0), # below tasmania
            Checkpoint(latitude=-15.668984, longitude=77.674694, radius=1190.0),
            Checkpoint(latitude=-39.438937, longitude=19.836265, radius=100.0),
            Checkpoint(latitude=14.881699, longitude=-21.024326, radius=100.0),
            Checkpoint(latitude=44.076538, longitude=-18.292936, radius=100.0),
            Checkpoint(
                latitude=config.start.latitude,
                longitude=config.start.longitude,
                radius=5,
            ),
        ]
        self.left = True
        self.counter = 0
        self.stuck_counter = 0
        self.stuck_course = 0
        
        self.old_position = Location(longitude=config.start.longitude, latitude=config.start.latitude)

    def run(
        self,
        t: float,
        dt: float,
        longitude: float,
        latitude: float,
        heading: float,
        speed: float,
        vector: np.ndarray,
        forecast: Callable,
        world_map: Callable,
    ) -> Instructions:
        """
        This is the method that will be called at every time step to get the
        instructions for the ship.

        Parameters
        ----------
        t:
            The current time in hours.
        dt:
            The time step in hours.
        longitude:
            The current longitude of the ship.
        latitude:
            The current latitude of the ship.
        heading:
            The current heading of the ship.
        speed:
            The current speed of the ship.
        vector:
            The current heading of the ship, expressed as a vector.
        forecast:
            Method to query the weather forecast for the next 5 days.
            Example:
            current_position_forecast = forecast(
                latitudes=latitude, longitudes=longitude, times=0
            )
        world_map:
            Method to query map of the world: 1 for sea, 0 for land.
            Example:
            current_position_terrain = world_map(
                latitudes=latitude, longitudes=longitude
            )

        Returns
        -------
        instructions:
            A set of instructions for the ship. This can be:
            - a Location to go to
            - a Heading to point to
            - a Vector to follow
            - a number of degrees to turn Left
            - a number of degrees to turn Right

            Optionally, a sail value between 0 and 1 can be set.
        """
        # Initialize the instructions
        instructions = Instructions()

        # ===========================================================

        # Go through all checkpoints and find the next one to reach
        for next_checkpoint in self.course:
            # Compute the distance to the checkpoint
            dist = distance_on_surface(
                longitude1=longitude,
                latitude1=latitude,
                longitude2=next_checkpoint.longitude,
                latitude2=next_checkpoint.latitude,
            )
            # Consider slowing down if the checkpoint is close
            jump = dt * np.linalg.norm(speed)
            if dist < 2.0 * next_checkpoint.radius + jump:
                instructions.sail = min(next_checkpoint.radius / jump, 1)
            else:
                instructions.sail = 1.0
            # Check if the checkpoint has been reached
            if dist < next_checkpoint.radius:
                next_checkpoint.reached = True
                self.set_new_course = True
            if not next_checkpoint.reached:
                break
        
        next_checkpoint_angle = goto(
                origin=Location(longitude=longitude, latitude=latitude),
                to=Location(longitude=next_checkpoint.longitude, latitude=next_checkpoint.latitude)
                )

        # correct for better wind usage
        u,v = forecast(
            latitudes=latitude, longitudes=longitude, times=0
        )
        wind_angle = self._get_angle([u, v])
        # adjust the course a little bit acording to winfd angle
        max_wind_angle = 50
        min_wind_angle = 135
        #print(f"{dt=}")
        if abs(next_checkpoint_angle-wind_angle) < max_wind_angle and next_checkpoint.radius > 50: # back wind
            if dist > 5.0 * next_checkpoint.radius or next_checkpoint.radius > 500:
                factor = 1 - (abs(next_checkpoint_angle-wind_angle) / max_wind_angle)
                course_angle = next_checkpoint_angle - (next_checkpoint_angle-wind_angle) * factor
                #print(f"{wind_angle=} {next_checkpoint_angle=} -> {course_angle=}")
            else:
                course_angle = next_checkpoint_angle
        elif abs(next_checkpoint_angle-wind_angle) > min_wind_angle and next_checkpoint.radius > 50: #front wind
            if dist > 2.0 * next_checkpoint.radius or next_checkpoint.radius > 500:
                factor = 1  - (min_wind_angle / abs(next_checkpoint_angle-wind_angle))
                if self.left:
                    course_angle = next_checkpoint_angle - (next_checkpoint_angle-wind_angle) * factor
                    self.counter += 1
                    if self.counter >= 5:
                        self.left = False
                        self.counter = 0
                else:
                    course_angle = next_checkpoint_angle + (next_checkpoint_angle-wind_angle) * factor
                    self.counter += 1
                    if self.counter >= 5:
                        self.left = True
                        self.counter = 0
            elif dist > 1.0 * next_checkpoint.radius: 
                factor = 1  - (min_wind_angle / abs(next_checkpoint_angle-wind_angle))
                if self.left:
                    course_angle = next_checkpoint_angle - (next_checkpoint_angle-wind_angle) * factor
                    self.counter += 1
                    if self.counter >= 2:
                        self.left = False
                        self.counter = 0
                else:
                    course_angle = next_checkpoint_angle + (next_checkpoint_angle-wind_angle) * factor
                    self.counter += 1
                    if self.counter >= 2:
                        self.left = True
                        self.counter = 0
                #print(f"{wind_angle=} {next_checkpoint_angle=} -> {course_angle=}")
            else:
                course_angle = next_checkpoint_angle
        else:
            course_angle = next_checkpoint_angle
            
        # colition detection 
        new_position = Location(longitude=longitude, latitude=latitude)
        water = self.old_position != new_position
        if not bool(water):
            if course_angle > 180:
                course_angle -= 135
            else:
                course_angle += 135

            self.stuck_course = course_angle
            self.stuck_counter += 1 
            #print(f"got stuck {self.stuck_counter} in  {water=} with new course {course_angle}")
        else:
            if self.stuck_counter:
                self.stuck_counter += 1
                course_angle = self.stuck_course
            if self.stuck_counter >= 10:
                self.stuck_counter = 0
         #+ wind_angle
            instructions.heading = Heading(course_angle)

        self.old_position = new_position
        return instructions

    def _get_angle(self, vec):
        vec = np.asarray(vec) / np.linalg.norm(vec)
        angle = np.arccos(np.dot(vec, [1, 0])) * 180 / np.pi
        if vec[1] < 0:
            angle = 360 - angle
        return angle

