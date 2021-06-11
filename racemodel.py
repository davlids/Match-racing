
import numpy as np
import random as rn
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import sys
import copy
import multiprocessing as mlp
mpl.use("TkAgg")

########################
### Helper functions ###
########################

def onclick(event):
    print("pause")
    plt.waitforbuttonpress()

def get_square_dist(p1, p2):
    return np.sum(np.square(p1 - p2))

def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def angle_between_vectors(v1, v2):

    dot_product = np.dot(v1, v2)
    return np.arccos(dot_product)

def rotate_vector(vec, radians):
    new_x = vec[0] * np.cos(radians) - vec[1] * np.sin(radians)
    new_y = vec[0] * np.sin(radians) + vec[1] * np.cos(radians)

    return np.round([new_x, new_y], 5)

def get_wind_dir(local_wind):
    v1 = [1, 0]
    v2 = unit_vector(local_wind)
    angle = angle_between_vectors(v1, v2)

    dir_state = np.round(12 * angle / np.pi)

    if v2[1] < 0:
        dir_state = 24 - dir_state

    return int(dir_state)

def get_unit_dir_2(dir):
    angle = dir_state_to_angle(dir, 24)
    unit_dir = np.array([np.cos(angle), np.sin(angle)])
    return unit_dir

def get_front_and_back_2(pos, dir, size):
    unit_dir = get_unit_dir_2(dir)
    front = pos + unit_dir * size / 2
    back = pos - unit_dir * size / 2

    return front, back

def check_collision(end_pos_0, end_pos_1, dir_0, dir_1, size):


    front_0, back_0 = get_front_and_back_2(end_pos_0, dir_0, size)
    front_1, back_1 = get_front_and_back_2(end_pos_1, dir_1, size)

    points_0 = [front_0, end_pos_0, back_0]
    points_1 = [front_1, end_pos_1, back_1]

    collision = False
    penalty = False
    
    for i in range(3):
        for j in range(i, 3):    
            dist = get_square_dist(points_0[i], points_1[j])

            if dist < 0.04:  
                return True, True        # [Penalty, collision]

            elif dist < 0.1: 
                penalty = True     # [Penalty, collision]
    

    return penalty, collision

def dir_state_to_angle(dir_state, n_directional_states):
    return dir_state * 2 * np.pi / n_directional_states

def get_velocity(speed, dir_state, n_directional_states):
    angle = dir_state_to_angle(dir_state, n_directional_states)
    return np.array(np.round([np.cos(angle), np.sin(angle)], 5)) * speed

def component_in_target_dir(speed, dir_state, target_v, n_directional_states):
    v = get_velocity(speed, dir_state, n_directional_states)
    unit_target = unit_vector(target_v)

    return np.dot(v, unit_target)

def clear_ahead(b0, b1):

    # Front and back of boats
    front_0, back_0 = b0.get_front_and_back()
    front_1, back_1 = b1.get_front_and_back()

    unit_dir_0 = b0.get_unit_dir()
    unit_dir_1 = b1.get_unit_dir()

    # Vector from back of boat 1 to front of boat 2
    back_to_front_0 = front_1 - back_0
    unit_back_to_front_0 = unit_vector(back_to_front_0)
    ahead_angle_0 = angle_between_vectors(unit_dir_0, unit_back_to_front_0)


    # Vector from back of boat 2 to front of boat 1
    back_to_front_1 = front_0 - back_1
    unit_back_to_front_1 = unit_vector(back_to_front_1)
    ahead_angle_1 = angle_between_vectors(unit_dir_1, unit_back_to_front_1)

    # Check if boat 1 is clear ahead
    if ahead_angle_0 > np.pi/2:
        return 0
    
    # Check if boat 2 is clear ahead
    elif ahead_angle_1 > np.pi/2:
        return 1

    else:
        return -1

def get_leeward_boat(b0, b1, on_starboard_tack):
    # On same tack and overlapping. Assuming wind in negativ y-dir
    front_0, back_0 = b0.get_front_and_back()
    front_1, back_1 = b1.get_front_and_back()

    if on_starboard_tack:
        if b0.get_x_pos() > b1.get_x_pos():
            return int(front_0[1] > back_1[1])
        else:
            return int(front_1[1] < back_0[1])
    else:
        if b0.get_x_pos() < b1.get_x_pos():
            return int(front_0[1] > back_1[1])
        else:
            return int(front_1[1] < back_0[1])

def right_of_way(b0, b1):   
    ## Assuming constant wind in negative y-dir
    # Is anybody tacking? Rule 13


    starboard_tack_0 = b0.get_tack()
    starboard_tack_1 = b1.get_tack()

    # If opposite tack, starboard has right of way. Rule 10
    if starboard_tack_0  and not starboard_tack_1:
        return 0

    if starboard_tack_1 and not starboard_tack_0:
        return 1

    # Else same tack, check overlap
    boat_ahead = clear_ahead(b0, b1)

    if boat_ahead == -1: #Rule 11
        return get_leeward_boat(b0, b1, starboard_tack_0)
        # return leeward boat

    else:  
        return boat_ahead

##################
### MCTS class ### 
##################


class MCTS:

    def __init__(self, time_to_start, time_step, n_options,parent = None, path = []):
        self.time_to_start = time_to_start
        self.time_step = time_step
        self.n_options = n_options
        self.path = np.array(path)
        self.parent = parent
        self.children = np.array([])
        self.w0 = 0  # Use s - w0 = w1
        self.s = 0

    def is_root(self):
        return self.parent == None

    def is_leaf(self):
        return len(self.children) == 0
    
    def is_fully_expanded(self):
        return len(self.children) == self.n_options ** 2

    def is_terminal(self):
        return self.time_to_start < 1

    def expand_all(self):

        for i in range(self.n_options):
            for j in range(self.n_options):
                tmp_path = np.vstack((self.path, [i, j]))
                self.children = np.append(self.children, MCTS(self.time_to_start - self.time_step, self.time_step, self.n_options,self, tmp_path))

    def expand_one(self):

        explored_options = [[child.path[-1,0], child.path[-1, 1]] for child in self.children]
        next_choice_options = np.array([[i,j] for i in range(self.n_options) for j in range(self.n_options) if [i, j] not in explored_options])
        c0, c1 = rn.choice(next_choice_options)
        new_path = np.vstack((self.path, [c0, c1]))
        new_child = MCTS(self.time_to_start - self.time_step, self.time_step, self.n_options,self, new_path)
        self.children = np.append(self.children, new_child)

        return new_child

    def DUCT_in_sim(self):
        c = 1.414213
        reward_sums_0 = np.zeros(self.n_options)
        reward_sums_1 = np.zeros(self.n_options)
        visit_sums_0 = np.zeros(self.n_options)
        visit_sums_1 = np.zeros(self.n_options)
        
        for child in self.get_children():
            ix_0 = child.path[-1, 0]
            ix_1 = child.path[-1, 1]

            reward_sums_0[ix_0] += child.w0
            visit_sums_0[ix_0] += child.s

            reward_sums_1[ix_1] += child.s - child.w0
            visit_sums_1[ix_1] += child.s

        uct_0 = np.array([reward_sums_0[i] / visit_sums_0[i] + c * np.sqrt( np.log(self.s) / visit_sums_0[i]) for i in range(self.n_options)])
        uct_1 = np.array([reward_sums_1[i] / visit_sums_1[i] + c * np.sqrt( np.log(self.s) / visit_sums_1[i]) for i in range(self.n_options)])

        chosen = np.array([np.argmax(uct_0), np.argmax(uct_1)])

        for child in self.get_children():
            if (chosen == child.path[-1]).all():
                return child

    def select(self):

        if self.is_terminal():
            return self

        elif not self.is_fully_expanded():
            return self.expand_one()

        else:
            selected_node = self.DUCT_in_sim()
            return selected_node.select()
        
    def backpropagate(self, winner):
        self.w0 += 1 - winner
        self.s += 1
        
        if not self.is_root():
            self.parent.backpropagate(winner)

    def get_simulation_path(self, b0, b1):
        seq_length = int(self.time_to_start / self.time_step) 
        seq_0 = np.append(self.path[1:,0], np.random.randint(2, size = seq_length))
        seq_1 = np.append(self.path[1:,1], np.random.randint(2, size = seq_length))

        b0.set_strat_sequence(seq_0)
        b1.set_strat_sequence(seq_1)
   
    def print_results(self):
        print(f'{self.w0}/{self.s}')

        for child in self.children:
            child.print_results()

    def get_children(self):
        return self.children


##################
### Race class ###
##################


class Race:

    def __init__(self, x_min, x_max, y_min, y_max, step_size, start_line, start_pos_0, start_pos_1, start_speed_0, start_speed_1,\
                 start_dir_0, start_dir_1, strat_0, strat_1,boat_size, n_directional_states, initial_wind, max_speed, acceleration_constant, dir_change_brake_factor,\
                 into_wind_brake_factor, n_simulation_steps, time_step, pause_time, plot_ix, n_sims, sel_met_0, sel_met_1):

        b0 = Boat(start_pos_0, start_speed_0, start_dir_0, boat_size, max_speed[0], acceleration_constant[0], dir_change_brake_factor[0],into_wind_brake_factor, n_directional_states, time_step, 0, strat_0, self, sel_met_0)
        b1 = Boat(start_pos_1, start_speed_1, start_dir_1, boat_size, max_speed[1], acceleration_constant[1], dir_change_brake_factor[1],into_wind_brake_factor, n_directional_states, time_step, 1, strat_1, self, sel_met_1)
        self.boats = [b0, b1]

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step_size = step_size
        self.n_steps_x = int((self.x_max - self.x_min + 1) / self.step_size)
        self.n_steps_y = int((self.y_max - self.y_min + 1) / self.step_size)
        self.x_mesh = np.linspace(x_min, x_max, self.n_steps_x)
        self.y_mesh = np.linspace(y_min, y_max, self.n_steps_y) 
        self.start_line = start_line 
        self.finish_line = np.array([[-1, 1], [self.y_max-2, self.y_max-2]])

        self.time_step = time_step
        self.pause_time = pause_time
        self.n_directional_states = n_directional_states
        self.n_simulation_steps = n_simulation_steps
        self.time_to_start = n_simulation_steps * time_step #n_simulation_steps
        self.run_time = 0
        self.initial_wind = initial_wind.copy()
        self.wind_grid = np.zeros((self.n_steps_y, self.n_steps_x, 2)) + initial_wind
        self.current_wind = initial_wind.copy()

        self.wind_shawdows = np.load('wind_shadow.npy')

        self.penalties = np.array([0 ,0])
        self.penalty_cooldown = 0

        self.plot_ix = plot_ix
        self.n_sims = n_sims
        self.est_utils = np.zeros((2, 4))

    def get_boats(self):
        return self.boats

    def get_x_min(self):
        return self.x_min
    
    def get_x_max(self):
        return self.x_max

    def get_y_min(self):
        return self.y_min

    def get_y_max(self):
        return self.y_max    

    def get_step_size(self):
        return self.step_size

    def get_n_steps_x(self):
        return int((self.x_max - self.x_min + 1) / self.step_size)

    def get_n_steps_y(self):
        return int((self.y_max - self.y_min + 1) / self.step_size)

    def get_wind(self):
        return self.wind_grid

    def get_wind_point(self, row, col):
        return self.wind_grid[row, col, :]

    def set_wind(self, new_wind):
        self.wind_grid = new_wind

    def set_new_wind(self):
        angle_change = (rn.random() - 0.5) * (np.pi / 180)
        speed_change = 1 + rn.random() / 50 - 0.01 
        self.current_wind = rotate_vector(self.current_wind, angle_change) * speed_change
        
    def reset_wind(self):
        self.set_new_wind()
        self.wind_grid = np.zeros((self.n_steps_y, self.n_steps_x, 2)) + self.current_wind

    def add_individual_wind_shadow(self, boat):

        shadow_ix = boat.get_wind_shadow_index()
        if shadow_ix == -1:
            return

        shadow = self.wind_shawdows[shadow_ix]
        shadow_size = len(shadow)
        new_wind = self.wind_grid.copy()
        grid_size_i, grid_size_j, _ = new_wind.shape
        grid_row, grid_col = boat.coordinates_to_grid_index()
        
        
        if shadow_ix == 0:
            start_row = grid_row
            start_col = grid_col - (shadow_size // 2) + 2

        elif shadow_ix == 1:
            start_row = grid_row + 1
            start_col = grid_col - (shadow_size // 2)

        elif shadow_ix == 2:
            start_row = grid_row - (shadow_size//2 + 1) + 1
            start_col = grid_col + 1

        elif shadow_ix == 3:
            start_row = grid_row - (shadow_size//2 + 1) + 1
            start_col = grid_col - shadow_size

        else:
            start_row = grid_row 
            start_col = grid_col - (shadow_size // 2) - 2
        

        for i in range(shadow_size):
            for j in range(shadow_size):
                tmp_i = start_row + i
                tmp_j = start_col + j
                if 0 <= tmp_i < grid_size_i and 0 <= tmp_j < grid_size_j:
                    new_wind[tmp_i][tmp_j] *= shadow[i][j]

        
        self.set_wind(new_wind)

    def update_wind_shadow(self):
        self.reset_wind()
        boats = self.get_boats()
        self.add_individual_wind_shadow(boats[0])
        self.add_individual_wind_shadow(boats[1])

    def give_penalty(self, boat):

        if self.run_time - self.penalty_cooldown > 8:
            self.penalties[boat] += 1
            self.penalties -= min(self.penalties)
            self.penalty_cooldown = self.run_time 
            #print(self.penalties)

    def get_finish_line(self):
        return self.finish_line

    def get_n_directional_states(self):
        return self.n_directional_states

    def get_time_to_start(self):
        return self.time_to_start

    def get_run_time(self):
        self.run_time
    
    def within_course(self, pos):
        if 1 < self.time_to_start < 120:
            return self.x_min < pos[0] < self.x_max and self.y_min < pos[1] < 0

        elif self.time_to_start < 120 and pos[1] < 0 and not (self.start_line[0, 0] + 0.2 < pos[0] < self.start_line[0, 1] - 0.2):
            return self.x_min < pos[0] < self.x_max and self.y_min < pos[1] < 0

        else:
            return self.x_min < pos[0] < self.x_max and self.y_min < pos[1] < self.y_max


    def run_race_save_plots(self):
        fig = plt.figure(figsize=(12, 8)) 
        fig.canvas.mpl_connect('button_press_event', onclick)
        gs = gridspec.GridSpec(1, 2,  width_ratios = [1.2, 1])
        ax = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1]) 
        ax.plot(self.start_line[0], self.start_line[1], 'r--')
        ax.plot(self.start_line[0][0], self.start_line[1][0], 'ro', markersize=8)
        ax.plot(self.start_line[0][1], self.start_line[1][1], 'ro', markersize=8)
        ax.plot(self.finish_line[0], self.finish_line[1], 'r--')
        ax.plot(self.finish_line[0][0], self.finish_line[1][0], 'ro', markersize=6)
        ax.plot(self.finish_line[0][1], self.finish_line[1][1], 'ro', markersize=6)
        ax.axis([self.x_min, self.x_max, self.y_min, self.y_max])
        ax.text(-11.5, self.finish_line[1 ,1] - 1, "Speed blue: ")
        ax.text(-11.5, self.finish_line[1 ,1] - 2.5, "Speed yellow: ")
        time_txt = ax.text(-11.5, self.finish_line[1 ,1] - 4, "Time to start: ")
        ax.text(8, self.finish_line[1 ,1] - 1, "Wind")
        ax.text(-11.7, 12.5, f' Penalties b, y : ')
        blue = (7 / 255, 129 / 255, 252 / 255)
        yellow = (255 / 255, 196 / 255, 0 / 255)

        center = [0, 6.5]   
        sq_size = 3
        dim = 4
        offset = 13
        left_x, right_x = center[0] - sq_size * dim / 2, center[0] + sq_size * dim / 2
        top_y, bottom_y = center[1] + sq_size * dim / 2, center[1] - sq_size * dim / 2

        sq_blue = np.array([[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y], [left_x, top_y]])
        sq_pol_blue = mpl.patches.Polygon(sq_blue, closed = True)
        sq_col_blue = [sq_pol_blue]
        coll_blue = mpl.collections.PatchCollection(sq_col_blue, zorder = 1, color = blue, alpha = 0.9)

        sq_yellow = np.array([[left_x, top_y], [right_x, top_y], [right_x, bottom_y], [left_x, bottom_y], [left_x, top_y]]) - [0, offset]
        sq_pol_yellow = mpl.patches.Polygon(sq_yellow, closed = True)
        sq_col_yellow = [sq_pol_yellow]
        coll_yellow = mpl.collections.PatchCollection(sq_col_yellow, zorder = 1, color = yellow, alpha = 0.9)

        tx_locs_x_blue = np.linspace(left_x + sq_size/2, right_x-sq_size/2, dim)
        tx_locs_y_blue = np.linspace(top_y - sq_size/2, bottom_y + sq_size/2, dim) 

        tx_locs_x_yellow = tx_locs_x_blue
        tx_locs_y_yellow = tx_locs_y_blue - offset


        for i in range(dim + 1):
            x_vert = left_x + i * sq_size
            y_horz = top_y - i * sq_size
            ax1.plot([x_vert, x_vert], [top_y, bottom_y], 'black', linewidth = 0.7)
            ax1.plot([left_x, right_x], [y_horz, y_horz], 'black', linewidth = 4)

            ax1.plot([x_vert, x_vert], [top_y - offset, bottom_y- offset], 'black', linewidth = 4)
            ax1.plot([left_x, right_x], [y_horz - offset, y_horz - offset], 'black', linewidth = 0.7)

        ax1.set_title('Estimated utilities')
        ax1.add_collection(coll_blue)
        ax1.add_collection(coll_yellow)
        ax_size = np.array([-8.8, 8.8, -13, 13])
        ax1.axis(ax_size)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
  
        trail_length = 30
        
        b0, b1 = self.get_boats()
        trail_0 = np.zeros((self.n_simulation_steps + trail_length + 100, 2)) 
        trail_0[:trail_length, :] = [b0.get_x_pos(), b0.get_y_pos()]
        trail_1 = np.zeros((self.n_simulation_steps + trail_length + 100, 2)) 
        trail_1[:trail_length] = [b1.get_x_pos(), b1.get_y_pos()]


        ### PRESTART LOOP####----------------------------------------------------------------------------------------------------------------------

        for step in range(self.n_simulation_steps):
            angle_b0 = b0.dir_state_to_angle_self()
            angle_b1 = b1.dir_state_to_angle_self()

            txt1 = ax.text(-6.7, self.finish_line[1 ,1] - 1, f"{b0.get_speed() * 10 :.1f} kts")
            txt2 = ax.text(-6.3, self.finish_line[1 ,1] -2.5, f"{b1.get_speed() * 10:.1f} kts")
            txt3 = ax.text(-6.3, self.finish_line[1 ,1] - 4, f"{self.time_to_start} s")
            txt4 = ax.text(-6, 12.5, f' {self.penalties}')
            plot_t0, = ax.plot(trail_0[step: step + trail_length, 0], trail_0[step: step + trail_length, 1], color = blue)
            plot_t1, = ax.plot(trail_1[step: step + trail_length, 0], trail_1[step: step + trail_length, 1], color = yellow)
            plot_b0 = ax.quiver(b0.get_x_pos(), b0.get_y_pos(), np.cos(angle_b0), np.sin(angle_b0),  scale = 40, color = blue, pivot = 'mid')
            plot_b1 = ax.quiver(b1.get_x_pos(), b1.get_y_pos(), np.cos(angle_b1), np.sin(angle_b1),  scale = 40, color = yellow, pivot = 'mid')
            wind_plot_1 = ax.quiver(8.1, self.finish_line[1 ,1] - 2, self.current_wind[0], self.current_wind[1], color = 'black', pivot = 'mid')
            wind_plot_2 = ax.text(8.8,self.finish_line[1 ,1] - 2, f"{np.linalg.norm(self.current_wind)*5:0.1f} m/s")
            
            txt_list = []
            for i in range(dim):
                for j in  range(dim):
                    w, n = b0.utils_list[:, i * dim + j]

                    tmp_tx = ax1.text(tx_locs_x_blue[j], tx_locs_y_blue[i], f'{100 * w/n:.0f} %', ha = 'center', va = 'center', fontsize = 12)
                    txt_list.append(tmp_tx)

                    w, n = b1.utils_list[:, i * dim + j]
                    tmp_tx = ax1.text(tx_locs_x_yellow[i], tx_locs_y_yellow[j], f'{100 * w/n:.0f} %', ha = 'center', va = 'center', fontsize = 12)
                    txt_list.append(tmp_tx)

            plt.savefig(f'./Race plots/Plots_{self.plot_ix}/step_{self.run_time}')
            txt1.remove()
            txt2.remove()
            txt3.remove()
            txt4.remove()
            plot_t0.remove()
            plot_t1.remove()
            plot_b0.remove()
            plot_b1.remove()
            wind_plot_1.remove()
            wind_plot_2.remove()

            for tx in txt_list:
                tx.remove() 


            self.update_wind_shadow()
            action_0 = b0.choose_next_action()
            action_1 = b1.choose_next_action()

            penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

            if penalty:
                priority = right_of_way(b0, b1)
                while True:
                    if priority == 0:
                        action_1 = b1.choose_next_action(True, collision)

                        if action_1 == 0:
                            if len(b1.get_penalty_options()) == 0:
                                print(f'Run ended process {self.plot_ix}')
                                return -1
                            else:
                                action_1 = b1.choose_penalty_action()
                                b1.reset_penalty_options()
                                self.give_penalty(1)
                                break
                                
                                
                    else:
                        action_0 = b0.choose_next_action(True, collision)

                        if action_0 == 0:
                            if len(b0.get_penalty_options()) == 0:
                                print(f'Run ended process {self.plot_ix}')
                                return -1
                            else:
                                action_0 = b0.choose_penalty_action()
                                b0.reset_penalty_options()
                                self.give_penalty(0)
                                break

                    penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

                    if not penalty:
                        break


            b0.update_boat(action_0)
            b1.update_boat(action_1)    

            if not self.within_course(b0.get_pos()):
                self.give_penalty(0)
            
            if not self.within_course(b1.get_pos()):
                self.give_penalty(1)

            b0.reset_previous_options()
            b1.reset_previous_options()

            b0.reset_penalty_options()
            b1.reset_penalty_options()

            if self.time_to_start % 4 == 0:
                self.sim_forward()

            trail_0[step + trail_length, :] = [b0.get_x_pos(), b0.get_y_pos()]
            trail_1[step + trail_length, :] = [b1.get_x_pos(), b1.get_y_pos()]
            
            self.time_to_start -= self.time_step
            self.run_time += self.time_step
            print(self.time_to_start)

        ### AFTER START LOOP ### --------------------------------------------------------------------------------------------------------------------------

        time_txt.remove()
        ax.text(-11.5, self.finish_line[1 ,1] - 4, "Race time: ")
        
        if not b0.strategy == b0.strategies['Go_to_finish_line']:
            b0.set_strategy('Go_to_finish_line')
            b0.set_new_target_list()
        if not b1.strategy == b1.strategies['Go_to_finish_line']:    
            b1.set_strategy('Go_to_finish_line')
            b1.set_new_target_list()

        for step in range(self.n_simulation_steps, self.n_simulation_steps + 300):
            angle_b0 = b0.dir_state_to_angle_self()
            angle_b1 = b1.dir_state_to_angle_self()
            txt1 = ax.text(-6.7, self.finish_line[1 ,1] - 1, f"{b0.get_speed() * 10 :.2f} kts")
            txt2 = ax.text(-6.3, self.finish_line[1 ,1] -2.5, f"{b1.get_speed() * 10:.2f} kts")
            txt3 = ax.text(-7, self.finish_line[1 ,1] - 4, f"{step * self.time_step - self.n_simulation_steps} s")
            txt4 = ax.text(-6, 12.5, f' {self.penalties}')
            plot_t0, = ax.plot(trail_0[step: step + trail_length, 0], trail_0[step: step + trail_length, 1], color = blue)
            plot_t1, = ax.plot(trail_1[step: step + trail_length, 0], trail_1[step: step + trail_length, 1], color = yellow)

            plot_b0 = ax.quiver(b0.get_x_pos(), b0.get_y_pos(), np.cos(angle_b0), np.sin(angle_b0),  scale = 35, color = blue, pivot = 'mid')
            plot_b1 = ax.quiver(b1.get_x_pos(), b1.get_y_pos(), np.cos(angle_b1), np.sin(angle_b1),  scale = 35, color = yellow, pivot = 'mid')
            wind_plot_1 = ax.quiver(8.1, self.finish_line[1 ,1] - 2, self.current_wind[0], self.current_wind[1], color = 'black', pivot = 'mid')
            wind_plot_2 = ax.text(8.8,self.finish_line[1 ,1] - 2, f"{np.linalg.norm(self.current_wind)*5:.1f} m/s")
            plt.savefig(f'./Race plots/Plots_{self.plot_ix}/step_{self.run_time}')
            txt1.remove()
            txt2.remove()
            txt3.remove()
            txt4.remove()
            plot_t0.remove()
            plot_t1.remove()
            plot_b0.remove()
            plot_b1.remove()
            wind_plot_1.remove()
            wind_plot_2.remove()

            self.update_wind_shadow()

            if self.penalties[0] > 0 and b0.get_y_pos() > 0 and not b0.is_taking_penalty():
                b0.set_taking_penalty()
                b0.set_penalty_start_dir(b0.get_dir_state())
                finished_dir = 8 - 4 * (6 < b0.get_dir_state() < 18) 
                b0.set_penalty_finished_dir(finished_dir)
                b0.set_strategy('Take_penalty')

            if self.penalties[1] > 0 and b1.get_y_pos() > 0 and not b1.is_taking_penalty():
                b1.set_taking_penalty()
                b1.set_penalty_start_dir(b1.get_dir_state())
                finished_dir = 8 - 4 * (6 < b1.get_dir_state() < 18) 
                b1.set_penalty_finished_dir(finished_dir)
                b1.set_strategy('Take_penalty')

            action_0 = b0.choose_next_action()
            action_1 = b1.choose_next_action()

            penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

            if penalty:
                priority = right_of_way(b0, b1)
                while True:
                    if priority == 0:
                        action_1 = b1.choose_next_action(True, collision)

                        if action_1 == 0:
                            if len(b1.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_1 = b1.choose_penalty_action()
                                b1.reset_penalty_options()
                                self.give_penalty(1)
                                break
                                
                                
                    else:
                        action_0 = b0.choose_next_action(True, collision)

                        if action_0 == 0:
                            if len(b0.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_0 = b0.choose_penalty_action()
                                b0.reset_penalty_options()
                                self.give_penalty(0)
                                break

                    penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

                    if not penalty:
                        break

            b0.update_boat(action_0)
            b1.update_boat(action_1)

            b0.reset_previous_options()
            b1.reset_previous_options()

            b0.reset_penalty_options()
            b1.reset_penalty_options()

            trail_0[step + trail_length, :] = [b0.get_x_pos(), b0.get_y_pos()]
            trail_1[step + trail_length, :] = [b1.get_x_pos(), b1.get_y_pos()]


            if b0.get_y_pos() > self.finish_line[1, 1]:
                txt1 = ax.text(-6.7, self.finish_line[1 ,1] - 1, f"{b0.get_speed() * 10 :.2f} kts")
                txt2 = ax.text(-6.3, self.finish_line[1 ,1] -2.5, f"{b1.get_speed() * 10:.2f} kts")
                txt3 = ax.text(-7, self.finish_line[1 ,1] - 4, f"{step * self.time_step - self.n_simulation_steps} s")
                plot_t0, = ax.plot(trail_0[step: step + trail_length, 0], trail_0[step: step + trail_length, 1], color = blue)
                plot_t1, = ax.plot(trail_1[step: step + trail_length, 0], trail_1[step: step + trail_length, 1], color = yellow)

                plot_b0 = ax.quiver(b0.get_x_pos(), b0.get_y_pos(), np.cos(angle_b0), np.sin(angle_b0),  scale = 35, color = blue, pivot = 'mid')
                plot_b1 = ax.quiver(b1.get_x_pos(), b1.get_y_pos(), np.cos(angle_b1), np.sin(angle_b1),  scale = 35, color = yellow, pivot = 'mid')
                wind_plot_1 = ax.quiver(8.1, self.finish_line[1 ,1] - 2, self.current_wind[0], self.current_wind[1], color = 'black', pivot = 'mid')
                wind_plot_2 = ax.text(8.8,self.finish_line[1 ,1] - 2, f"{np.linalg.norm(self.current_wind)*5:.1f} m/s") 
                plt.savefig(f'./Race plots/Plots_{self.plot_ix}/step_{self.run_time}')
                return 0
            elif b1.get_y_pos() > self.finish_line[1, 1]:
                txt1 = ax.text(-6.7, self.finish_line[1 ,1] - 1, f"{b0.get_speed() * 10 :.2f} kts")
                txt2 = ax.text(-6.3, self.finish_line[1 ,1] -2.5, f"{b1.get_speed() * 10:.2f} kts")
                txt3 = ax.text(-7, self.finish_line[1 ,1] - 4, f"{step * self.time_step - self.n_simulation_steps} s")
                plot_t0, = ax.plot(trail_0[step: step + trail_length, 0], trail_0[step: step + trail_length, 1], color = blue)
                plot_t1, = ax.plot(trail_1[step: step + trail_length, 0], trail_1[step: step + trail_length, 1], color = yellow)

                plot_b0 = ax.quiver(b0.get_x_pos(), b0.get_y_pos(), np.cos(angle_b0), np.sin(angle_b0),  scale = 35, color = blue, pivot = 'mid')
                plot_b1 = ax.quiver(b1.get_x_pos(), b1.get_y_pos(), np.cos(angle_b1), np.sin(angle_b1),  scale = 35, color = yellow, pivot = 'mid')
                wind_plot_1 = ax.quiver(8.1, self.finish_line[1 ,1] - 2, self.current_wind[0], self.current_wind[1], color = 'black', pivot = 'mid')
                wind_plot_2 = ax.text(8.8,self.finish_line[1 ,1] - 2, f"{np.linalg.norm(self.current_wind)*5:.1f} m/s") 
                plt.savefig(f'./Race plots/Plots_{self.plot_ix}/step_{self.run_time}')
                return 1
            

            self.run_time += self.time_step                

    def run_race_no_plots(self):
        b0, b1 = self.get_boats()

        for step in range(self.n_simulation_steps):

            self.update_wind_shadow()
            action_0 = b0.choose_next_action()
            action_1 = b1.choose_next_action()

            penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

            if penalty:
                priority = right_of_way(b0, b1)
                while True:
                    if priority == 0:
                        action_1 = b1.choose_next_action(True, collision)

                        if action_1 == 0:
                            if len(b1.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_1 = b1.choose_penalty_action()
                                b1.reset_penalty_options()
                                self.give_penalty(1)
                                break
                                
                                
                    else:
                        action_0 = b0.choose_next_action(True, collision)

                        if action_0 == 0:
                            if len(b0.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_0 = b0.choose_penalty_action()
                                b0.reset_penalty_options()
                                self.give_penalty(0)
                                break

                    penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

                    if not penalty:
                        break


            b0.update_boat(action_0)
            b1.update_boat(action_1)    

            if not self.within_course(b0.get_pos()):
                self.give_penalty(0)
            
            if not self.within_course(b1.get_pos()):
                self.give_penalty(1)

            b0.reset_previous_options()
            b1.reset_previous_options()

            b0.reset_penalty_options()
            b1.reset_penalty_options()

            if self.time_to_start % 4 == 0:
                self.sim_forward()

            self.time_to_start -= self.time_step
            self.run_time += self.time_step
            print(self.time_to_start)


        if not b0.strategy == b0.strategies['Go_to_finish_line']:
            b0.set_strategy('Go_to_finish_line')
            b0.set_new_target_list()
        if not b1.strategy == b1.strategies['Go_to_finish_line']:    
            b1.set_strategy('Go_to_finish_line')
            b1.set_new_target_list()

        for step in range(self.n_simulation_steps, self.n_simulation_steps + 300):

            self.update_wind_shadow()

            if self.penalties[0] > 0 and b0.get_y_pos() > 0 and not b0.is_taking_penalty():
                b0.set_taking_penalty()
                b0.set_penalty_start_dir(b0.get_dir_state())
                finished_dir = 8 - 4 * (6 < b0.get_dir_state() < 18) 
                b0.set_penalty_finished_dir(finished_dir)
                b0.set_strategy('Take_penalty')

            if self.penalties[1] > 0 and b1.get_y_pos() > 0 and not b1.is_taking_penalty():
                b1.set_taking_penalty()
                b1.set_penalty_start_dir(b1.get_dir_state())
                finished_dir = 8 - 4 * (6 < b1.get_dir_state() < 18) 
                b1.set_penalty_finished_dir(finished_dir)
                b1.set_strategy('Take_penalty')

            action_0 = b0.choose_next_action()
            action_1 = b1.choose_next_action()

            penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

            if penalty:
                priority = right_of_way(b0, b1)
                while True:
                    if priority == 0:
                        action_1 = b1.choose_next_action(True, collision)

                        if action_1 == 0:
                            if len(b1.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_1 = b1.choose_penalty_action()
                                b1.reset_penalty_options()
                                self.give_penalty(1)
                                break
                                
                                
                    else:
                        action_0 = b0.choose_next_action(True, collision)

                        if action_0 == 0:
                            if len(b0.get_penalty_options()) == 0:
                                return -1
                            else:
                                action_0 = b0.choose_penalty_action()
                                b0.reset_penalty_options()
                                self.give_penalty(0)
                                break

                    penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], b0.get_size())

                    if not penalty:
                        break

            b0.update_boat(action_0)
            b1.update_boat(action_1)

            b0.reset_previous_options()
            b1.reset_previous_options()

            b0.reset_penalty_options()
            b1.reset_penalty_options()


            if b0.get_y_pos() > self.finish_line[1, 1]:
                #Blue won
                print(b0.strat_weights, b1.selection_method)
                return 0
            elif b1.get_y_pos() > self.finish_line[1, 1]:
                #Yellow won
                print(b0.strat_weights, b1.selection_method)
                return 1
            

            self.run_time += self.time_step 


    def sim_to_start(self, current_boat, other_boat, time_between_dec, current_boat_ix):
        other_boat_ix = ( current_boat_ix + 1 ) % 2

        ix = 0
        while self.time_to_start > 0:

            if self.time_to_start % time_between_dec == 0:
                current_boat.set_strategy_sim(current_boat.strat_sequence[ix])
                other_boat.set_strategy_sim(other_boat.strat_sequence[ix])
                ix += 1
    

            action_0 = current_boat.choose_next_action()
            action_1 = other_boat.choose_next_action()

            penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], current_boat.get_size())

            if penalty:
                priority = right_of_way(current_boat, other_boat)
                while True:
                    if priority == 0:
                        action_1 = other_boat.choose_next_action(True, collision)

                        if action_1 == 0:
                            if len(other_boat.get_penalty_options()) == 0:
                                #Collision
                                return 0
                            else:
                                action_1 = other_boat.choose_penalty_action()
                                other_boat.reset_penalty_options()
                                self.give_penalty(other_boat_ix)
                                break
                                
                                
                    else:
                        action_0 = current_boat.choose_next_action(True, collision)

                        if action_0 == 0:
                            if len(current_boat.get_penalty_options()) == 0:
                                #Collision
                                return 1
                            else:
                                action_0 = current_boat.choose_penalty_action()
                                current_boat.reset_penalty_options()
                                self.give_penalty(current_boat_ix)
                                break

                    penalty, collision = check_collision(action_0[0], action_1[0], action_0[2], action_1[2], current_boat.get_size())

                    if not penalty:
                        break

            current_boat.update_boat(action_0)
            other_boat.update_boat(action_1)
            
            if not self.within_course(current_boat.get_pos()):
                self.give_penalty(current_boat_ix)
            
            if not self.within_course(other_boat.get_pos()):
                self.give_penalty(other_boat_ix)

            current_boat.reset_previous_options()
            other_boat.reset_previous_options()

            current_boat.reset_penalty_options()
            other_boat.reset_penalty_options()

            self.time_to_start -= self.time_step
            self.run_time += self.time_step
        
        return -1

    def sim_post_start(self, current_boat, other_boat, current_ix):
        other_ix = ( current_ix + 1 ) % 2
        if not current_boat.strategy == current_boat.strategies['Go_to_finish_line']:
            current_boat.set_strategy('Go_to_finish_line')
            current_boat.set_new_target_list()
        if not other_boat.strategy == other_boat.strategies['Go_to_finish_line']:    
            other_boat.set_strategy('Go_to_finish_line')
            other_boat.set_new_target_list()

        for _ in range(500):

            if self.penalties[current_ix] > 0 and current_boat.get_y_pos() > 0 and not current_boat.is_taking_penalty():
                current_boat.set_taking_penalty()
                current_boat.set_penalty_start_dir(current_boat.get_dir_state())
                finished_dir = 8 - 4 * (6 < current_boat.get_dir_state() < 18) 
                current_boat.set_penalty_finished_dir(finished_dir)
                current_boat.set_strategy('Take_penalty')
                self.penalties[current_ix] -= 1

            if self.penalties[other_ix] > 0 and other_boat.get_y_pos() > 0 and not other_boat.is_taking_penalty():
                other_boat.set_taking_penalty()
                other_boat.set_penalty_start_dir(other_boat.get_dir_state())
                finished_dir = 8 - 4 * (6 < other_boat.get_dir_state() < 18) 
                other_boat.set_penalty_finished_dir(finished_dir)
                other_boat.set_strategy('Take_penalty')
                self.penalties[other_ix] -= 1


            action_0 = current_boat.choose_next_action()
            action_1 = other_boat.choose_next_action()

            current_boat.update_boat(action_0)
            other_boat.update_boat(action_1)

            current_boat.reset_previous_options()
            other_boat.reset_previous_options()

            current_boat.reset_penalty_options()
            other_boat.reset_penalty_options()

            if current_boat.get_y_pos() > self.finish_line[1, 1]:
                # Current boat won
                return 0

            elif other_boat.get_y_pos() > self.finish_line[1, 1]:
                # Other boat won
                return 1

        #print("Sim not finished!")
        return -1

    def multi_process_sim_forward(self, boat_ix, time_between_dec, n_strats, ret_list):
        sim_race = copy.deepcopy(self)
        sim_boats = sim_race.get_boats()
        sim_race.reset_wind()
        current_boat = sim_boats[boat_ix]
        other_boat = sim_boats[(boat_ix + 1) % 2]

        root = MCTS(sim_race.time_to_start, time_between_dec, n_strats , path = [0, 0]) 

        for _ in range(sim_race.n_sims):
            
            #Expand next step
            tmp_node = root.select()
            tmp_node.get_simulation_path(current_boat, other_boat)
            winner = sim_race.sim_to_start(current_boat, other_boat, time_between_dec, boat_ix)# Simulate from end node untill race is finished
            
            if winner == -1:
                winner = sim_race.sim_post_start(current_boat, other_boat, boat_ix)

            if winner >= 0:
                tmp_node.backpropagate(winner)
            
            # Resetting to initial values 
            sim_race = copy.deepcopy(self)
            sim_boats = sim_race.get_boats()
            current_boat = sim_boats[boat_ix]
            other_boat = sim_boats[(boat_ix + 1) % 2]

        #ret_list.append(current_boat.choose_best_strat_sim(root.get_children(), n_strats))
        ret_list.append(root.get_children())
        
    def sim_forward(self):
        n_strats = 4
        time_between_dec = 8

        manager = mlp.Manager()
        ret_list_0 = manager.list()
        ret_list_1 = manager.list()
        p0 = mlp.Process(target=self.multi_process_sim_forward , args = [0, time_between_dec, n_strats, ret_list_0])
        p1 = mlp.Process(target=self.multi_process_sim_forward , args = [1,  time_between_dec, n_strats, ret_list_1])
        p0.start()
        p1.start()
        p0.join()
        p1.join()

        b0, b1 = self.get_boats()
        b0.set_utils(ret_list_0[0], n_strats)
        b1.set_utils(ret_list_1[0], n_strats)
        b0.choose_obj_MCTS(n_strats)
        b1.choose_obj_MCTS(n_strats)

    
##################
### Boat class ###
##################


class Boat:

    def __init__(self, pos, speed, dir_state, size, max_speed, acceleration_constant, dir_change_brake_factor, into_wind_brake_factor, n_directional_states, time_step, ix, strategy,race, selection_method):
        self.pos = pos
        self.speed = speed
        self.dir_state = dir_state
        self.size = size
        self.race = race
        self.time_step = time_step

        self.ix = ix
        self.max_speed = max_speed
        self.acceleration_constant = acceleration_constant
        self.dir_change_brake_factor = dir_change_brake_factor
        self.into_wind_brake_factor = into_wind_brake_factor

        self.n_directional_states = n_directional_states

        self.go_to_target = False
        self.target = None
        self.previous_choice = None
        self.previous_options = None
        self.strategies = self.strats_to_dict()
        self.strategy = self.strategies[strategy]
        self.back_up_strategy = self.strategy
        self.selection_method = selection_method

        self.penalty_options = np.array([])
        self.target_point_list = None
        self.target_list_index = 0

        self.penalty_start_dir = 0
        self.penalty_finished_dir = 0
        self.taking_penalty = False

        self.utils_list = np.ones((2, 16))
        self.last_choice = 0
        self.prediction_list = np.zeros(3)
        self.strat_weights = np.zeros(3)

    def strats_to_dict(self):
        strategies = {}
        #strategies['Random'] = self.random_strat
        strategies['Stay_between'] = self.stay_between_start_strat
        strategies['Go_to_target'] = self.go_to_target_strat
        strategies['Follow'] = self.follow_strat
        strategies['Circle'] = self.circle_strat
        strategies['Go_to_finish_line'] = self.get_to_finish_line
        strategies['Take_penalty'] = self.take_penalty_strat
        return strategies

    def get_pos(self):
        return self.pos

    def get_x_pos(self):
        return self.pos[0]

    def get_y_pos(self):
        return self.pos[1]

    def get_speed(self):
        return self.speed
    
    def get_dir_state(self):
        return self.dir_state

    def get_tack(self):
        return 6 < self.dir_state < 18
        # 1 if starboard, 0 if port

    def get_size(self):
        return self.size
    
    def get_unit_dir(self):
        angle = self.dir_state_to_angle_self()
        unit_dir = np.array([np.cos(angle), np.sin(angle)])
        return unit_dir

    def get_front_and_back(self):
        unit_dir = self.get_unit_dir()
        front = self.pos + unit_dir * self.size / 2
        back = self.pos - unit_dir * self.size / 2

        return front, back

    def set_pos(self, new_pos):
        self.pos = new_pos

    def set_speed(self, new_speed):
        self.speed = new_speed
    
    def set_dir_state(self, new_dir):
        self.dir_state = new_dir

    def set_penalty_start_dir(self, dir):
        self.penalty_start_dir = dir

    def set_penalty_finished_dir(self, dir):
        self.penalty_finished_dir = dir

    def set_strat_sequence(self, seq):
        self.strat_sequence = seq

    def is_taking_penalty(self):
        return self.taking_penalty

    def set_taking_penalty(self):
        self.taking_penalty = True

    def coordinates_to_grid_index(self):
        col = int(((self.get_x_pos() - self.race.get_x_min()) / (self.race.get_x_max() - self.race.get_x_min()) ) * self.race.get_n_steps_x())
        row = self.race.get_n_steps_y() - int(((self.get_y_pos() - self.race.get_y_min()) / (self.race.get_y_max() - self.race.get_y_min()) ) * self.race.get_n_steps_y()) - 1

        return row, col
        
    def get_n_directional_states(self):
        return self.n_directional_states

    def dir_state_to_angle_self(self):
        return self.dir_state * 2 * np.pi / self.n_directional_states

    def get_acceleration_constant(self):
        return self.acceleration_constant

    def get_max_speed(self):
        return self.max_speed

    def get_dir_change_brake_factor(self):
        return self.dir_change_brake_factor

    def get_into_wind_brake_factor(self):
        return self.into_wind_brake_factor()

    def get_wind_shadow_index(self):
        dir_state = self.dir_state
    
        if 2 <= dir_state <= 4:
            return 4

        elif 5 <= dir_state <= 7:
            return -1

        elif 8 <= dir_state <= 11:
            return 0

        elif 14 <= dir_state <= 18:
            return 2
        
        elif 19 <= dir_state <= 22:
            return 3
        
        else:
            return 1

    def get_velocity_self(self):
        angle = self.dir_state_to_angle_self()
        return np.array(np.round([np.cos(angle), np.sin(angle)], 5)) * self.speed
    
    def reset_penalty_options(self):
        self.penalty_options = np.array([])

    def get_penalty_options(self):
        return self.penalty_options

    def get_slow_down_options(self, dir_state, speed):
        grid_row, grid_col = self.coordinates_to_grid_index()
        grid_row = max(min(grid_row, self.race.get_n_steps_y() - 1), 0) # Adjusting if oob
        grid_col = max(min(grid_col, self.race.get_n_steps_x() - 1), 0)

        local_wind = self.race.get_wind_point(grid_row, grid_col)
        wind_dir_state = get_wind_dir(local_wind)
        rel_wind_dir = np.abs(wind_dir_state - dir_state)
        options = [1]


        if 4 < rel_wind_dir < 11:
            options.append(0.8)

        elif speed > 0.8 * self.max_speed:
            options.append(0.9)

        return options

    def tack_dir_adjustment(self, new_dir_state):

        if 4 <= self.dir_state <= 7 and 8 <= new_dir_state <= 9:
            return 2 + (9 - new_dir_state)

        elif 5 <= self.dir_state <= 8 and 3 <= new_dir_state <= 4:
            return -2  - (4 - new_dir_state)
        
        else:
            return 0

    def new_speed(self, dir_state_change, speed_inp = None):

        if speed_inp == None:
            current_speed = self.speed
        else:
            current_speed = speed_inp

        grid_row, grid_col = self.coordinates_to_grid_index()
        grid_row = max(min(grid_row, self.race.get_n_steps_y() - 1), 0) # Adjusting if oob
        grid_col = max(min(grid_col, self.race.get_n_steps_x() - 1), 0)

        local_wind = self.race.get_wind_point(grid_row, grid_col)
        local_wind_strength = np.linalg.norm(local_wind)

        wind_dir_state = get_wind_dir(local_wind)
        new_dir_state = (self.dir_state + dir_state_change) % self.n_directional_states

        rel_wind_dir = np.abs(wind_dir_state - new_dir_state)
        if rel_wind_dir < 3 or rel_wind_dir == 10:
            tmp_max_speed = self.max_speed * 0.7
        else:
            tmp_max_speed = self.max_speed

        if 10 < rel_wind_dir < 14: # If 24 states
            if current_speed > 0.2:
                return current_speed * (self.into_wind_brake_factor / local_wind_strength) ** self.time_step
            else:
                return max(current_speed - 0.05, -0.05)
        
        elif dir_state_change == 0:
            return min(current_speed + self.acceleration_constant*local_wind_strength*self.time_step, tmp_max_speed * local_wind_strength)

        else:
            return min( max(0.0, current_speed * (self.dir_change_brake_factor ** np.abs(dir_state_change))), tmp_max_speed * local_wind_strength)

    def find_next_action_options(self):
        
        # Add something to handle dial up
        high_speed = self.speed > 0.7 * self.max_speed # Limit ability to change direction if high speed
        
        dir_change_options = [i for i in range(-self.time_step*2 + high_speed, self.time_step*2 + 1 - high_speed)]
        options = []

        for dir_state_change in dir_change_options:
            tmp_dir_state = (self.dir_state + dir_state_change) % self.n_directional_states
            tmp_new_speed = self.new_speed(dir_state_change)
            slow_down_options = self.get_slow_down_options(tmp_dir_state, tmp_new_speed)

            for speed_factor in slow_down_options:
                tmp_new_speed_2 = tmp_new_speed * speed_factor
                tmp_new_pos = self.pos + get_velocity(tmp_new_speed_2, tmp_dir_state, self.n_directional_states) * self.time_step

                if self.race.within_course(tmp_new_pos):
                    tmp_dir_state += self.tack_dir_adjustment(tmp_dir_state) 
                    tmp_option = [tmp_new_pos, tmp_new_speed_2, tmp_dir_state]
                    options.append(tmp_option)

                else:
                    oob_option = np.array([tmp_new_pos, tmp_new_speed_2, tmp_dir_state], dtype=object)

                    if len(self.penalty_options) == 0:
                        self.penalty_options = oob_option
                    else:
                        self.penalty_options = np.vstack((self.penalty_options, oob_option))

      
        return np.array(options, dtype=object)

    def check_if_dead_end(self, decision):
        pos = decision[0]
        speed = decision[1]
        dir_state = decision[2]
        
        high_speed = speed > 0.6 * self.max_speed
        dir_change_options = [i for i in range(-self.time_step*2 + high_speed, self.time_step*2 + 1 - high_speed)]

        for dir_state_change in dir_change_options:
            tmp_dir_state = (dir_state + dir_state_change) % self.n_directional_states
            tmp_new_speed = self.new_speed(dir_state_change, speed) * 0.5
            tmp_new_pos = pos + get_velocity(tmp_new_speed, tmp_dir_state, self.n_directional_states) * self.time_step

            if self.race.within_course(tmp_new_pos):
                return False
        
        return True

    def stay_between_start_strat(self, options):
    
        other_boat = self.race.get_boats()[((self.ix + 1) % 2)]
        pos_other = other_boat.get_pos()
        rel_pos_factor = 0.6
        target_point = pos_other * rel_pos_factor
        target_vec = target_point - self.pos
        chosen = np.argmax([component_in_target_dir(options[i, 1], options[i, 2], target_vec, self.n_directional_states) for i in range(options.shape[0])]) 


        return chosen

    def random_strat(self, options):
        return rn.randint(0, options.shape[0]-1)

    def go_to_target_strat(self, options):
        target_vec = self.target - self.pos

        #chosen = np.argmax([component_in_target_dir(options[i, 1], options[i, 2], target_vec, self.n_directional_states) for i in range(options.shape[0]) if options[i,1] > -0.01]) 
        chosen = np.argmax([component_in_target_dir(options[i, 1], options[i, 2], target_vec, self.n_directional_states) for i in range(options.shape[0])])

        return chosen


    def follow_strat(self, options):

        other_boat = self.race.get_boats()[((self.ix + 1) % 2)]
        _, other_back = other_boat.get_front_and_back()
        other_dir_state = other_boat.get_dir_state()
        self.target = (other_back - other_boat.get_unit_dir()*0.8) * 0.75

        if get_square_dist(self.pos, other_back) > 1:
            chosen = self.go_to_target_strat(options)
        
        elif np.abs(self.dir_state - other_dir_state) > 4:
            chosen = np.argmin(np.abs(options[:,2] - other_boat.get_dir_state()))

        else:
            chosen = np.argmin(np.abs(options[:,1] - other_boat.get_speed()))
        
        return chosen
   
    def take_penalty_strat(self, options):

        if self.dir_state == self.penalty_finished_dir:
            self.taking_penalty = False
            self.set_strategy('Go_to_finish_line')
            self.set_new_target_list()
            self.race.penalties[self.ix] -= 1
            return self.strategy(options)

        else:
            if self.penalty_finished_dir == 4:
                chosen = np.argmin((4 - options[:, 2]) % 24)
            else:
                chosen = np.argmin((options[:, 2] - 8) % 24)
            
            return chosen
    
    def circle_strat(self, options):
        self.target_point_list = np.array([[8, -8], [7, -2], [7, -7], [5, -3]])

        if get_square_dist(self.pos, self.target_point_list[self.target_list_index]) < 0.2:
            self.target_list_index = (self.target_list_index + 1) % 4

        self.target = self.target_point_list[self.target_list_index]
        chosen =  self.go_to_target_strat(options)
        
        return chosen

    def get_to_finish_line(self, options):
        
        if get_square_dist(self.pos, self.target) < 0.3:
            self.target_list_index += 1
            self.target = self.target_point_list[self.target_list_index]
            #print("New target!", self.ix, self.target)

        return self.go_to_target_strat(options)

        
    def find_path_to_finish_line(self, current_pos):
        start_left = current_pos[0] > 0
        finish_line = self.race.get_finish_line()

        dy = finish_line[1, 1] - current_pos[1]
        dx =  (2 * start_left - 1) * current_pos[0]

        l1 = (dy + dx) / 2
        l2 = (dy - dx) / 2

        if start_left:  
            x_start_line_target = current_pos[0] + current_pos[1]
            point_1 = current_pos + [-l1, l1]
            point_2 = point_1 + [l2, l2]
            point_3 = point_2 + [0, 1]
        
        else:
            x_start_line_target = current_pos[0] - current_pos[1]
            point_1 = current_pos + [l1 , l1]
            point_2 = point_1 + [-l2, l2]
            point_3 = point_2 + [0, 1]

        point_0 = np.array([x_start_line_target, 0])
        aim_points = np.array([point_0, point_1, point_2, point_3])
        #aim_points = np.array([point_2, point_3])
        return aim_points

    def find_path_to_finish_line_after_start(self):
        current_pos = self.pos
        start_left = current_pos[0] > 0
        finish_line = self.race.get_finish_line()
        
        dy = finish_line[1, 1] - current_pos[1]
        dx =  (2 * start_left - 1) * current_pos[0]

        if start_left:
            no_tack_x_target = current_pos[0] - dy
            no_tack_needed = finish_line[0, 0] < no_tack_x_target
            x_target = min(no_tack_x_target, finish_line [0, 0])
        else:
            no_tack_x_target = current_pos[0] + dy
            no_tack_needed = no_tack_x_target < finish_line[0, 1]
            x_target = min(no_tack_x_target, finish_line [1, 0])

        if no_tack_needed:
            point_1 = np.array([x_target, finish_line[1, 0]])
            point_2 = point_1 + [0, 1]
            
            aim_points =  np.array([point_1, point_2])
            
        else:
            l1 = (dy + dx) / 2
            l2 = (dy - dx) / 2

            if start_left:  
                point_1 = current_pos + [-l1, l1]
                point_2 = point_1 + [l2, l2]
                point_3 = point_2 + [0, 1]
            
            else:
                point_1 = current_pos + [l1 , l1]
                point_2 = point_1 + [-l2, l2]
                point_3 = point_2 + [0, 1]


            aim_points = np.array([point_1, point_2, point_3])

        return aim_points

    def set_target_list_pre(self):
        go_right = self.dir_state < 6 or self.dir_state > 18
        x = self.get_x_pos()
        y = self.get_y_pos()

        if x < 0:
            zone_fact = x - y

            if zone_fact < -2.8:  # in zone 0   --->  go to zone 1 (straight right)
                aim_points = np.array([y-2.5, y])
                aim_points = np.vstack((aim_points, self.find_path_to_finish_line(aim_points)))
                

            elif zone_fact < 2.8: # in zone 1
                aim_points = self.find_path_to_finish_line(self.pos)
                

            else: # in zone 2
                if go_right:
                    diff = - (3 + y + x) / 2 + 0.5 # adjustment factor
                    int_point = [x + diff, y + diff]
                else:
                    diff =  (3 + y - x) / 2 - 0.5 # adjustment factor
                    int_point = [x + diff, y - diff]

                
                aim_points = np.array(int_point)
                aim_points = np.vstack((aim_points, self.find_path_to_finish_line(aim_points)))

            self.target_point_list = aim_points

        else:  # x > 0
            zone_fact = x + y
            if zone_fact  > 2.8:    #in zone 4   ---->  go to zone 3 (straight left)
                aim_points = np.array([-y+2.5, y])
                aim_points = np.vstack((aim_points, self.find_path_to_finish_line(aim_points)))

            elif zone_fact > -2.8: # in zone 3
                aim_points = self.find_path_to_finish_line(self.pos)
               
            else: # in zone 2
                if go_right:
                    diff = - (3 + y + x) / 2 + 0.5 # adjustment factor
                    int_point = [x + diff, y + diff] # fix this point
                else:
                    diff =  (3 + y - x) / 2 - 0.5 # adjustment factor
                    int_point = [x + diff, y - diff] # fix this point

                
                aim_points = np.array(int_point)
                aim_points = np.vstack((aim_points, self.find_path_to_finish_line(aim_points)))
                   
            self.target_point_list = aim_points

    def set_new_target_list(self):

        if self.pos[1] < 0:
            self.set_target_list_pre()
            self.target = self.target_point_list[0]
            self.target_list_index = 0
        
        else:
            self.target_point_list = self.find_path_to_finish_line_after_start()
            self.target = self.target_point_list[0]
            self.target_list_index = 0

    def is_close_leeward(self):
        other_boat = self.race.get_boats()[((self.ix + 1) % 2)]
        both_on_starboard = self.get_tack() and other_boat.get_tack()
        both_on_port = not self.get_tack() and not other_boat.get_tack()
        sq_dist = get_square_dist(self.pos, other_boat.get_pos())
        is_leeward = not get_leeward_boat(self, other_boat, both_on_starboard) 

        return (both_on_starboard or both_on_port) and is_leeward and sq_dist < 1.5
    
    def choose_penalty_action(self):

        if len(self.penalty_options.shape) == 1:
            return  self.penalty_options

        chosen_option_ix = self.strategy(self.penalty_options)

        return self.penalty_options[chosen_option_ix, :]

    def choose_next_action(self, give_way = False, collision = False):

        if give_way:
            options = self.previous_options
            if not collision:
                if len(self.penalty_options) == 0:
                    self.penalty_options = options[self.previous_choice, :].copy()
                else:
                    self.penalty_options = np.vstack((self.penalty_options, options[self.previous_choice, :]))

                
            options = np.delete(options, self.previous_choice, 0)

            if len(options) < 1:
                return 0
                
        else:
            options = self.find_next_action_options()

        if len(options) > 0:
            chosen_option_ix = self.strategy(options)
            self.previous_choice = chosen_option_ix
            self.previous_options = options

            return options[chosen_option_ix, :]

        elif len(self.penalty_options) > 0:
            chosen_option_ix = self.strategy(self.penalty_options)
            self.previous_choice = chosen_option_ix
            self.previous_options = self.penalty_options

            return self.penalty_options[chosen_option_ix, :]

        else:    
            print("Out of options")
            print(self.pos)
            print(self.find_next_action_options())
            print(self.penalty_options)
            sys.exit()

    def update_boat(self, chosen_option):
        self.set_pos(chosen_option[0])
        self.set_speed(chosen_option[1])
        self.set_dir_state(chosen_option[2])

    def set_strategy_sim(self, ix):
        if ix == 0:
            self.set_strategy('Stay_between')

        elif ix == 1:
            self.set_strategy('Follow')

        elif ix == 2:
            self.set_strategy('Circle')

        elif ix == 3:
            if self.strategies['Go_to_finish_line'] == self.strategy:
                return
            else:
                self.set_strategy('Go_to_finish_line')
                self.set_new_target_list()

    def set_strategy(self, strat):
        self.strategy = self.strategies[strat]

    def reset_previous_options(self):
        self.previous_options = None
        self.previous_choice = None

    def choose_best_strat_sim(self, nodes, n_options):
        reward_sums = np.zeros(n_options)
        visit_sums = np.zeros(n_options)
        
        for node in nodes:
            ix = node.path[-1, 0]

            reward_sums[ix] += node.w0
            visit_sums[ix] += node.s

        duct = reward_sums / visit_sums
        best_strat = np.argmax(duct)
        return np.append(best_strat, duct)  

    def set_utils(self, nodes, n_strats):

        for node in nodes:
            ix = node.path[-1, 0] * n_strats + node.path[-1, 1]
            self.utils_list[0, ix] = node.w0
            self.utils_list[1, ix] = node.s

    def choose_obj_MCTS(self, n_strats):
        best_strat = 0
        if self.selection_method == 'DUCT':
            duct = np.array([sum(self.utils_list[0, i*n_strats : i*n_strats+n_strats]) / sum(self.utils_list[1, i*n_strats:i*n_strats+n_strats]) for i in range(n_strats)])
            best_strat = np.argmax(duct)
            
        elif self.selection_method == 'Maxmin': # Maxmin
                
            est_utils = self.utils_list[0, :] / self.utils_list[1, :]
            worst_case = np.array([ min(est_utils[i*n_strats : i*n_strats+n_strats]) for i in range(n_strats)])
            best_strat = np.argmax(worst_case)

        elif self.selection_method == 'Maxmax':
            est_utils = self.utils_list[0, :] / self.utils_list[1, :]
            best_case = np.array([ max(est_utils[i*n_strats : i*n_strats+n_strats]) for i in range(n_strats)])
            best_strat = np.argmax(best_case)

        elif self.selection_method == 'Exploit':
            self.choose_obj_predict(n_strats)
            return

        self.last_choice = best_strat
        self.set_strategy_sim(best_strat)

    def choose_obj_predict(self, n_strats):
        other_boat = self.race.get_boats()[((self.ix + 1) % 2)]

        # Update weigts based on previous choice
        for i in range(len(self.prediction_list)):
            if self.prediction_list[i] == other_boat.last_choice:
                self.strat_weights[i] += 1
        
        # Predict next choice

        # Higherst avg
        opponent_inv_duct = np.array([sum(self.utils_list[0, i:i + (n_strats-1)*n_strats + 1:n_strats]) / sum(self.utils_list[1, i:i + (n_strats-1)*n_strats + 1:n_strats]) for i in range(n_strats)])
        opponent_pred_strat_0 = np.argmin(opponent_inv_duct)
        self.prediction_list[0] = opponent_pred_strat_0

        # Maxmin
        opponent_inv_maxmin = np.array( [ max(self.utils_list[0, i:i + (n_strats-1)*n_strats + 1:n_strats] ) for i in range(n_strats)]  )
        opponent_pred_strat_1 = np.argmin(opponent_inv_maxmin)
        self.prediction_list[1] = opponent_pred_strat_1

        # Maxmax
        opponent_inv_maxmax = np.array( [ min(self.utils_list[0, i:i + (n_strats-1)*n_strats + 1:n_strats] ) for i in range(n_strats)]  )
        opponent_pred_strat_2 = np.argmin(opponent_inv_maxmax)
        self.prediction_list[2] = opponent_pred_strat_2
        

        predicted_strat = np.argmax(self.strat_weights)
        predicted_action = int(self.prediction_list[predicted_strat])
        est_utils = self.utils_list[0, :] / self.utils_list[1, :]
        best_strat = np.argmax( est_utils[predicted_action : n_strats * (n_strats-1) + predicted_action + 1 : n_strats] )

        self.set_strategy_sim(best_strat)

