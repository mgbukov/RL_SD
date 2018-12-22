from .model import MODEL
import numpy as np

class SD:
    def __init__(self, param, model:MODEL, nflip = 1, init_random=False):
        from itertools import combinations
        self.n_step = param['n_step']
        self.model = model

        idx = np.arange(0,self.n_step,dtype=int)

        if nflip == 1:
            self.flip_list = np.arange(0, self.n_step, dtype=int)
        else:
            self.flip_list = [i for i in range(self.n_step)] # list of idx
            for s in range(1, nflip):
                for e in combinations(idx,s+1):
                    self.flip_list.append(list(e))
        self.n_move = len(self.flip_list)
        self.order = np.arange(0, self.n_move, dtype=int)
        self.fid_series = param['fid_series']
        self.init_random=init_random

    def run(self):
        """ Perform SD descent"""

        if self.init_random is True: # random initialization
            self.model.update_protocol(np.random.randint(0, self.model.n_h_field, size=self.n_step))

        if self.fid_series is True: # storing fidelity traces (can slow down things quite a bit, so left as option !)
            return self.run_with_fid_series()
        else:
            return self.run_wo_fid_series()

    def run_with_fid_series(self):
        model = self.model
        old_fid = model.compute_fidelity()
        fid_series = [old_fid]
        n_fid_eval = 1
        n_visit = 1
        move_history = []
        local_minima_reached = False

        while not local_minima_reached:

            np.random.shuffle(self.order)

            for move in self.order:
                idx_flip = self.flip_list[move]
                model.flip_hx(idx_flip)
                new_fid = model.compute_fidelity()
                n_fid_eval +=1
                if new_fid > old_fid:
                    old_fid = new_fid
                    fid_series.append(old_fid)
                    move_history.append(idx_flip)
                    n_visit += 1
                    break
                else:
                    model.flip_hx(idx_flip) # reject move

                if move == self.order[-1]: # meaning it went through the whole sequence !
                    local_minima_reached = True
      
        return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit, fid_series, move_history

    def run_wo_fid_series(self):
        
        model = self.model
        old_fid = model.compute_fidelity()
        n_fid_eval = 1
        n_visit = 1
        local_minima_reached = False

        while not local_minima_reached:

            np.random.shuffle(self.order)

            for move in self.order:
                idx_flip = self.flip_list[move]
                model.flip_hx(idx_flip)
                new_fid = model.compute_fidelity()
                n_fid_eval +=1
                if new_fid > old_fid:
                    old_fid = new_fid
                    n_visit += 1
                    break
                else:
                    model.flip_hx(idx_flip) # reject move

                if move == self.order[-1]: # meaning it went through the whole sequence !
                    local_minima_reached = True
      
        return old_fid, np.copy(model.protocol()), n_fid_eval, n_visit, [-1], [-1]