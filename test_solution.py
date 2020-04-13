#!/usr/bin/env python3

from misio.optilio.lost_wumpus import run_agent
from misio.lost_wumpus.testing import test_locally
from misio.lost_wumpus.agents import RandomAgent, SnakeAgent, AgentStub
from misio.lost_wumpus._wumpus import Action, Field
import numpy as np

np.set_printoptions(precision=3, suppress=True)
n = 10


class MyAgent(AgentStub):
    def __init__(self, *args, **kwargs):
        super(MyAgent, self).__init__(*args, **kwargs)
        self.map_exit = np.argwhere(self.map == Field.EXIT)[0]
        self.histogram = np.ones_like(self.map, dtype=np.float64)
        self.normalize()

    def reset(self):
        self.histogram = np.ones_like(self.map, dtype=np.float64)
        self.normalize()

    def normalize(self):
        array_sum = np.sum(self.histogram)
        self.histogram = self.histogram / array_sum

    def closest(self, where):
        rows = where[0][0] - self.map_exit[0]
        columns = where[0][1] - self.map_exit[1]

        rows = rows if abs(rows) < (self.h - abs(rows)) else -rows
        columns = columns if abs(columns) < (
            self.w - abs(columns)) else -columns

        distance = rows + columns
        location = where[0]

        for place in where:
            rows = place[0] - self.map_exit[0]
            columns = place[1] - self.map_exit[1]

            rows = rows if abs(rows) < (self.h - abs(rows)) else -rows
            columns = columns if abs(columns) < (
                self.w - abs(columns)) else -columns

            temp = rows + columns

            if temp < distance:
                distance = temp
                location = place

        return location

    def calculate_move(self, move_type):
        new_histogram = np.multiply(self.histogram, (1.0 - self.p) / 4.0)

        if Action.UP:
            moved_histogram = np.roll(self.histogram, -1, axis=0)
            new_histogram = np.add(new_histogram, np.multiply(
                moved_histogram, self.p))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(self.histogram, -2, axis=0), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, -1, axis=1), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, 1, axis=1), (1.0 - self.p) / 4.0))
        elif Action.DOWN:
            moved_histogram = np.roll(self.histogram, 1, axis=0)
            new_histogram = np.add(new_histogram, np.multiply(
                moved_histogram, self.p))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(self.histogram, 2, axis=0), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, -1, axis=1), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, 1, axis=1), (1.0 - self.p) / 4.0))
        elif Action.LEFT:
            moved_histogram = np.roll(self.histogram, -1, axis=1)
            new_histogram = np.add(new_histogram, np.multiply(
                moved_histogram, self.p))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(self.histogram, -2, axis=1), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, -1, axis=0), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, 1, axis=0), (1.0 - self.p) / 4.0))
        elif Action.RIGHT:
            moved_histogram = np.roll(self.histogram, 1, axis=1)
            new_histogram = np.add(new_histogram, np.multiply(
                moved_histogram, self.p))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(self.histogram, 2, axis=1), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, -1, axis=0), (1.0 - self.p) / 4.0))
            new_histogram = np.add(new_histogram, np.multiply(
                np.roll(moved_histogram, 1, axis=0), (1.0 - self.p) / 4.0))

        self.histogram = new_histogram

    def sense(self, sensory_input: bool):
        if sensory_input:
            for index, field in np.ndenumerate(self.histogram):
                self.histogram[index[0], index[1]] = field * \
                    self.pj if self.map[index[0], index[1]
                                        ] == Field.CAVE else field * self.pn
        self.normalize()

    def move(self):
        wumpus = np.argwhere(self.histogram == np.max(self.histogram))
        wumpus = self.closest(wumpus)

        rows = wumpus[0] - self.map_exit[0]
        cols = wumpus[1] - self.map_exit[1]

        rows = rows if abs(rows) < (self.h - abs(rows)) else -rows
        cols = cols if abs(cols) < (self.w - abs(cols)) else -cols

        action = Action.UP

        if abs(rows) > abs(cols):
            if rows > 0:
                action = Action.UP
            elif rows < 0:
                action = Action.DOWN
            else:
                if cols > 0:
                    action = Action.LEFT
                else:
                    action = Action.RIGHT
        else:
            if cols > 0:
                action = Action.LEFT
            elif cols < 0:
                action = Action.RIGHT
            else:
                if rows > 0:
                    action = Action.UP
                else:
                    action = Action.DOWN

        self.calculate_move(action)

        return action

    def get_histogram(self):
        return self.histogram


test_locally("tests/2015.in", RandomAgent, n=n)
# test_locally("tests/2016.in", RandomAgent, n=n)

test_locally("tests/2015.in", SnakeAgent, n=n)
# test_locally("tests/2016.in", SnakeAgent, n=n)

test_locally("tests/2015.in", MyAgent, n=n)
# test_locally("tests/2016.in", MyAgent, n=n)

# run_agent(MyAgent)
