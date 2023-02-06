import pyNN.spiNNaker as sim
import numpy as np
import matplotlib.pyplot as plt
import pyNN.utility.plotting as plt_2s
from pyNN.random import RandomDistribution
import random
import matplotlib.cm as cm
from pyNN.random import NumpyRNG
from spynnaker.pyNN.models.neuron.builds import IFCondExpBase
from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexNeuronsSynapses)


excit_param = {'tau_m': 100.0,
               'cm': 100.0,
               'v_rest': -65.0,
               'v_reset': -65.0,
               'tau_refrac': 5.0,
               'i_offset': 0,
               'v_thresh': -53.0,
               }
inhit_param = {'tau_m': 10.0,
               'cm': 10.0,
               'v_rest': -60.0,
               'v_reset': -45.0,
               'v_thresh': -40.0,
               'tau_syn_E': 1.0,
               'tau_syn_I': 2.0,
               'tau_refrac': 2.0,
               'i_offset': 0,
               'e_rev_E': 0,
               'e_rev_I': -85
               }
t_one_sample = 500
## create network
chosenCmap = cm.get_cmap('hot_r')
ie_conn = np.ones((9, 9))
for i in range(9):
    ie_conn[i, i] = 0


def weight_extraction_to_numpy_1_d(pre_n, post_n, target_projec):
    raw_weight_from_conn = target_projec.get(['weight'], 'list')
    raw_weight_form = np.zeros((pre_n, post_n))
    for m in range(pre_n):
        for n in range(post_n):
            raw_weight_form[m, n] = raw_weight_from_conn[m * post_n + n][-1]

    return raw_weight_form


def weight_extraction_to_numpy(pre_n, post_n, target_projec):
    if type(target_projec) is list:
        raw_weight_form = np.zeros((len(target_projec), pre_n, post_n))
        for index, item in enumerate(target_projec):
            raw_weight_form[index] = weight_extraction_to_numpy_1_d(pre_n, post_n, item)
    else:
        raw_weight_form = weight_extraction_to_numpy_1_d(pre_n, post_n, target_projec)

    return raw_weight_form

def give_rearranged_weight(pre_num, post_num, raw_weight_form):
    pattern_num = int(np.sqrt(post_num))
    pix_num_each_pattern = int(np.sqrt(pre_num))
    rearrange_size = pattern_num * pix_num_each_pattern
    rearranged_weight = np.zeros((rearrange_size, rearrange_size))

    for i in range(pattern_num):
        for j in range(pattern_num):
            rearranged_weight[i * pix_num_each_pattern:(i + 1) * pix_num_each_pattern,
            j * pix_num_each_pattern:(j + 1) * pix_num_each_pattern] = raw_weight_form[:, i * pattern_num + j].reshape(
                (pix_num_each_pattern, pix_num_each_pattern))

    return rearranged_weight

def make_network(rate, start_time,duration ):
    source_pop_obj = sim.extra_models.SpikeSourcePoissonVariable(rates=rate, starts=start_time, durations=duration)

    post_spliter = SplitterAbstractPopulationVertexNeuronsSynapses(1)

    source_pop = sim.Population(25,source_pop_obj)

    excit_pop = sim.Population(9, IFCondExpBase(**excit_param), label='excitatory',
                               additional_parameters={'splitter': post_spliter})
    excit_pop.initialize(v=-105)

    inhit_pop = sim.Population(9, sim.IF_cond_exp(**inhit_param), label='inhibitory')
    inhit_pop.initialize(v=-100)

    timing_stdp = sim.extra_models.PfisterSpikeTriplet(tau_plus=20, tau_minus=20, tau_x=40, tau_y=40, A_plus=0,
                                                     A_minus=0.0001)
    weight_stdp = sim.extra_models.WeightDependenceAdditiveTriplet(w_min=0, w_max=1.0, A3_plus=0.01, A3_minus=0)

    numpy_RNG = NumpyRNG(seed=42)

    stdp_model_triplet = sim.STDPMechanism(timing_dependence=timing_stdp, weight_dependence=weight_stdp,
                                         weight=RandomDistribution('normal', mu=0.1, sigma=0.1, rng=numpy_RNG),
                                         delay=RandomDistribution('uniform', (1, 5), rng=numpy_RNG))
    input_projec = sim.Projection(source_pop, excit_pop, sim.AllToAllConnector(), synapse_type=stdp_model_triplet,
                                receptor_type="excitatory")

    E2I_projec = sim.Projection(excit_pop, inhit_pop, sim.OneToOneConnector(),
                              synapse_type=sim.StaticSynapse(weight=10.4))
    I2E_projec = sim.Projection(inhit_pop, excit_pop, sim.ArrayConnector(ie_conn), receptor_type='inhibitory',
                              synapse_type=sim.StaticSynapse(weight=17))

    return excit_pop, inhit_pop, input_projec


sim.setup(1)

rate_gen_np = np.load('./small_data_gen/noisy_data.npy')
rate_gen_trans = np.transpose(rate_gen_np)
start_time = [500*i for i in range(40)]
duration = [350]*40
excit_pop, inhit_pop, stdp_projec = make_network(rate_gen_trans[:, :40],start_time, duration)
excit_pop.record('spikes')

sim.run(t_one_sample*40)

weight = weight_extraction_to_numpy(25, 9, stdp_projec)
rearrange_w = give_rearranged_weight(25, 9, weight)
spike = excit_pop.get_data('spikes')



fig = plt.figure(1)
ax = plt.gca()
plt_2s.plot_spiketrains(ax, spike.segments[0].spiketrains, yticks=True, xticks=True)

fig2 = plt.figure(2)
im = plt.imshow(rearrange_w, interpolation="nearest", vmin=0, cmap=chosenCmap)
plt.colorbar(im)
plt.title('weights of connection ')
fig2.canvas.draw()

plt.show()

sim.end()







