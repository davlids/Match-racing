import numpy as np
import time
import racemodel
import concurrent.futures


def run_parallel_sims(plot_ix):
        x_min = -12
        x_max = 12
        y_min = -10
        y_max = 20
        step_size = 0.1 
        start_line = np.array([[-3.2, 3.2], [0, 0]])

        strat_0 = 'Follow' # Blue boat (port)
        strat_1 = 'Circle' # Yellow boat (starboard)

        n_directional_states = 24
        boat_size = 0.8
        time_step = 1
        pause_time = 0.6
        max_speed = [0.8, 0.8]
        acceleration_constant = [0.1, 0.1]
        dir_change_brake_factor = [0.95, 0.95]
        into_wind_brake_factor = 0.5
        initial_wind = [0, -1]

        start_pos_0 = np.array([np.random.rand() * 7, np.random.rand() * 3 - 4])   
        start_pos_1 = np.array([np.random.rand() * 7, np.random.rand() * 3 - 4])
        start_speed_0 = np.random.rand() / 2.5 + 0.1
        start_speed_1 = np.random.rand() / 2.5 + 0.1
        start_dir_0 = np.random.randint(15, high = 22)
        start_dir_1 = np.random.randint(15, high = 22)

        n_sims = 1000
        n_simulation_steps = 120

        selection_method_0 = 'Exploit'
        selection_method_1 = 'Maxmin'
        
        race = racemodel.Race(x_min, x_max, y_min, y_max, step_size, start_line, start_pos_0, start_pos_1, start_speed_0, start_speed_1,\
                        start_dir_0, start_dir_1, strat_0, strat_1,boat_size, n_directional_states, initial_wind, max_speed, acceleration_constant, dir_change_brake_factor,\
                        into_wind_brake_factor, n_simulation_steps, time_step, pause_time, plot_ix, n_sims, selection_method_0, selection_method_1)

        winner = race.run_race_no_plots()

        return winner


if __name__ == "__main__":

    start = time.time()

    n_processes = 6
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(run_parallel_sims, i) for i in range(n_processes)]

    for result in results:
        print(result.result())


    stop = time.time()
    run_time = stop - start
    print(f'Time: {run_time}')