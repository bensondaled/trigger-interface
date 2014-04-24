from tsc_exp_analyze import analysis1, analysis2
from tsc_exp_analyze import Analysis

sigma = 1.5
datadir = '/Volumes/LACIE SHARE/Tsc1_Social'
gr1 = ['Tsc1_103', 'Tsc1_105', 'Tsc1_223-80', 'Tsc1_240-107', 'Tsc1_248-135', 'Tsc1_260-152', 'Tsc1_261-154', 'Tsc1_276-161', 'Tsc1_279-174','Tsc1_223-86']
gr2 = ['Tsc1_221-63', 'Tsc1_248-138-2', 'Tsc1_258-144', 'Tsc1_261-151']
gr3 = ['Tsc1_104', 'Tsc1_219-145', 'Tsc1_228-95', 'Tsc1_228-99', 'Tsc1_240-109', 'Tsc1_242-115', 'Tsc1_242-116', 'Tsc1_248-132', 'Tsc1_260-149', 'Tsc1_260-153']

#analysis1(mice=gr1, datadir=datadir, sigma=sigma, norm=True, show_perc=True)
analysis2(mice=gr1+gr2+gr3, datadir=datadir)
