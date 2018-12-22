import numpy as np
from matplotlib import pyplot as plt
from qsp_opt import QSP
from qsp_opt import plotting
import sys, os
import pickle
from copy import deepcopy


''' plt.rc('text', usetex=True)
font = {'family' : 'serif', 'size'   : 40}
plt.rc('font', **font)
import matplotlib as mpl
from ml_style import ml_style_1 as sty
mpl.rcParams.update(sty.style)
mpl.rcParams['font.size']=20 '''

def load_data(file_name):
    #print(file_name)
    _, data = pickle.load(open(file_name,'rb'))
    return data

def load_data_par(model, T=2.0, n_step=40, n_flip=1):
    model.parameters['T'] = T
    model.parameters['n_step']= n_step
    model.parameters['dt']= model.parameters['T']/model.parameters['n_step']
    model.parameters['n_flip']=n_flip
    root = "../data/"
    file_name = root + model.make_file_name()
    return load_data(file_name), file_name


def load_best_protocol(n_flip=4, n_step=80, trange = None, root = ""):

    if trange is None:
        trange = np.arange(0.1,4.01,0.1)
    argv = ['','n_step=%i'%n_step,'n_flip=%i'%n_flip,'n_partition=%i'%int(n_step/10)]
    model = QSP(argv)
    best_prot = {}

    for t in trange:
        model.parameters['T'] = t
        model.parameters['dt'] = t/n_step
        fname = root + model.make_file_name()
        _, data = pickle.load(open(fname,'rb'))
        print(t,'\t',len(data))
        fid = [d[1] for d in data]
        pos_best_fid = np.argmax(fid)
        best_prot[int(t*10)] = [data[pos_best_fid][1], data[pos_best_fid][4]]

    return best_prot

def load_excitation(n_step=40):
    return pickle.load(open('DOS/exc_dict_vs_T_nStep=%i_L=6.pkl'%n_step,'rb'))

def average_excite(trange, prot_vs_T, n_step):
    from itertools import combinations
    delta1_vs_T = {}
    delta2_vs_T = {}
    for t in trange:
        _, prot = prot_vs_T[int(t*10)]
        sf1 = np.arange(0, n_step)
        sf2 = list(combinations(sf1, 2))
        argv = ['T=%.2f'%t,'n_step=%i'%n_step,'n_partition=%i'%int(n_step/10)]
        model = QSP(argv = argv)

        delta_SF1 = []
        delta_SF2 = []

        init_F = model.evaluate_protocol(prot)
        init_MAG = np.sum(2*prot-1)

        for idx in sf1:
            # 1 SF
            prot[idx]^=1
            final_F_1 = model.evaluate_protocol(prot)
            delta_SF1.append(final_F_1 - init_F)
            prot[idx]^=1

        # 2 SF
        for idx in sf2:
            sf2_idx = np.array(idx)
            prot[sf2_idx]^=1
            final_MAG = np.sum(2*prot-1)
            if abs(final_MAG - init_MAG) < 1e-6: # mag = 0 excitations 
                final_F_2 = model.evaluate_protocol(prot)
                delta_SF2.append(final_F_2 - init_F)
            prot[sf2_idx]^=1

        delta1_vs_T[int(t*10)] = np.mean(delta_SF1)
        delta2_vs_T[int(t*10)] = np.mean(delta_SF2)

    return delta1_vs_T, delta2_vs_T 

def main():

    trange = np.arange(0.1,4.01,0.1)
    exc_dict40 = load_excitation(n_step=40)
    exc_dict60 = load_excitation(n_step=60)
    exc_dict80 = load_excitation(n_step=80)
    mean_exc = {40:[],60:[],80:[]}
    for t in trange:
        mean_exc[40].append(np.mean(exc_dict40[int(t*10)][1][:,0]))
        mean_exc[60].append(np.mean(exc_dict60[int(t*10)][1][:,0]))
        mean_exc[80].append(np.mean(exc_dict80[int(t*10)][1][:,0]))

    for n in [40,60,80]:
        plt.plot(trange,np.array(mean_exc[n])/(n**1.8))
    plt.show()
    print(exc_dict[10][1])
    exit()

    # -> from a random protocol ! (see average scaling)
    # check excitations for 1 SF (M=2). Check excitations for 2SF -> only M=0 excitations)
    #n_step = 80
    n_sample = 3000
    trange = np.arange(0.1,4.01,0.1)
    ''' d1 = {}
    d2 = {}
    best_prot40 = load_best_protocol(n_flip=4,n_step=40,root="../data/")
    d1[40],d2[40]= average_excite(trange,best_prot40,40)
    best_prot60 = load_best_protocol(n_flip=4,n_step=60,root="../data/")
    d1[60],d2[60]= average_excite(trange,best_prot60,60)
    best_prot80 = load_best_protocol(n_flip=4,n_step=80,root="../data/")
    d1[80],d2[80]= average_excite(trange,best_prot80,80)
    with open('tmp.pkl','wb') as f:
        pickle.dump([d1,d2],f) '''

    [d1,d2] = pickle.load(open('tmp.pkl','rb'))
    #print(d1[40])

    for n in [40,60,80]:
        plt.plot(trange, [d2[n][int(t*10)]*n**0.5 for t in trange])
    plt.show()
    exit()

    nrange =  [50, 100, 150, 200]
    res1_vs_T = {}
    res2_vs_T = {}

    ''' for t in trange:
        print(t)
        res1 = {}
        res2 = {}
        for n_step in nrange:
            sf1 = np.arange(0, n_step)
            sf2 = list(combinations(sf1, 2))
            argv = ['T=%.2f'%t,'n_step=%i'%n_step,'n_partition=%i'%int(n_step/10)]
            model = QSP(argv = argv)

            delta_SF1 = []
            delta_SF2 = []
            for _ in range(n_sample):
                rp = np.random.randint(0, 2, n_step)
                init_F = model.evaluate_protocol(rp)
                init_MAG = np.sum(2*rp-1)

                # 1 SF
                rsf1_idx = np.random.randint(0, n_step)
                rp[rsf1_idx]^=1
                final_F_1 = model.evaluate_protocol(rp)

                delta_SF1.append(final_F_1 - init_F)
                rp[rsf1_idx]^=1

                # 2 SF
                while True:
                    tmp = np.random.randint(0,len(sf2))
                    rsf2_idx =  np.array(sf2[tmp])    
                    rp[rsf2_idx]^=1
        

                    final_MAG = np.sum(2*rp-1)
                    if abs(final_MAG - init_MAG) < 1e-6: # mag = 0 excitations 
                        final_F_2 = model.evaluate_protocol(rp)
                        delta_SF2.append(final_F_2 - init_F)
                        break

            res1[n_step] = np.mean(np.abs(delta_SF1))
            res2[n_step] = np.mean(np.abs(delta_SF2))
        res1_vs_T[int(t*10)] = deepcopy(res1)
        res2_vs_T[int(t*10)] = deepcopy(res2)

    with open('tmp_res.pkl','wb') as f:
        pickle.dump([res1_vs_T, res2_vs_T],f) '''
    
    with open('tmp_res.pkl','rb') as f:
        [res1_vs_T, res2_vs_T] = pickle.load(f)

    n_res_1 = {50:[],100:[],150:[],200:[]}
    n_res_2 = {50:[],100:[],150:[],200:[]}

    for n in nrange:
        for t in trange:
            n_res_1[n].append(res1_vs_T[int(t*10)][n])
            n_res_2[n].append(res2_vs_T[int(t*10)][n])

    for n in nrange:
        plt.plot(trange, np.array(n_res_1[n])*n**0.88)
    plt.show()
    exit()
    for n in nrange:
        plt.plot(trange, np.array(n_res_2[n])*n**(0.48))

    plt.title('Exc 1')
    plt.show()
    exit()

    n50_res = []
    n100_res = []
    n150_res = []
    exit()

    
    plt.scatter(nrange, mean_res1)
    plt.scatter(nrange, mean_res2)
    plt.show()

    exit()






    





    #_, data = load_data_par(model, T=2.0, n_step=40)

    ''' for L in [1,6]:
        for n_flip in [1,2,3,4]:
            q_function(L=L, n_flip=n_flip, show=False) '''
    ''' model = QSP(argv = sys.argv.append("L=1"))
    print(model.make_file_name())
    exit()
    data, _ = load_data_par(model,T=1.9,n_step=60, n_flip=4)
    n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
    fid = np.array(fid)
    print(len(fid))
    plt.hist(fid)
    plt.show()
    #plt.hist(fid[fid<0.768])
    #print(len(fid[fid>0.768]))
    #plt.hist(fid[fid<0.7,bins=70)
    #plt.show()
    exit()

    print(data[0][4])
    r=np.random.randint(0,200)
    p=data[r][4]
    plotting.protocol(p,T=1.9,title="$F=%.6f$"%data[r][1])
    exit()
    #plotting.protocol(
     '''
    fig1(fontsize=16, n_step=60, L=6, show=True)
    


    exit()

def excitations(L=1):
    """ relative excitations from local minimas ! """
    argv = sys.argv
    argv.append("L=%i"%L)
    model = QSP(argv = argv)

    """ n_eval, fid, energy, n_visit, protocol """

    mean_fid = np.zeros(40)
    max_fid = np.zeros(40)
    mean_exc=np.zeros(40)
    #std_fid = np.zeros(40)
    qEA = np.zeros(40)
    Trange = np.arange(0.1,4.01,0.1)
    n_step = 40

    #for n_step in [20,24,28,32,36,40]:
    #for n_flip in [1,2,3,4]:
    
    n_flip=1
    for i, T in enumerate(Trange):
        data, _ = load_data_par(model,T=T,n_step=n_step, n_flip=n_flip)
        n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
        mean_exc[i], _ = mean_excitation(model, protocol)

    plt.scatter(Trange, mean_exc)
    plt.tight_layout(w_pad=0.5)
    plt.show()

    
def fig1(fontsize=16, n_step=40, L=6, show=True):
    """ q vs T plots """

    argv = sys.argv
    argv.append("L=%i"%L)

    model = QSP(argv = argv)

    """ n_eval, fid, energy, n_visit, protocol """


    Trange = np.arange(0.1,4.01,0.1)
    Trange = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.3,3.4,3.5,3.6,3.7]
    mean_fid = np.zeros(len(Trange))
    qEA = np.zeros(len(Trange))
    fid = {}

    for n_flip in [1,2,3,4]:
        for i, T in enumerate(Trange):
            data, _ = load_data_par(model,T=T,n_step=n_step, n_flip=n_flip)
            n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
            mean_fid[i] = np.max(fid)
            qEA[i] = qfunction(protocol)
        fid[n_flip] = np.copy(mean_fid)
        #plt.scatter(Trange,qEA,c="k")
        plt.plot(Trange, qEA,label="$SD%i$"%n_flip,marker='o', lw=1)

    plt.plot(Trange,fid[4],'-.',c='k',ms=4,lw=1.5,label="$F_h$")
    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('$T$')
    plt.ylabel('$q(T)$')
    plt.tight_layout(pad=0.2)

    plt.savefig('plots/q_L=%i_nStep=%i.pdf'%(L,n_step))

    if show is True:
        plt.show()
    plt.clf()

def fig1b(fontsize=16, n_step=40, L=6, show=True):
    argv = sys.argv
    argv.append("L=%i"%L)

    from sklearn.decomposition import PCA
    from tsne_visual import TSNE

    model = QSP(argv = argv)

    """ n_eval, fid, energy, n_visit, protocol """


    Trange = np.arange(0.1,4.01,0.1)
    Trange = [0.1,0.4,0.7,1.0,1.3,1.6,1.7,2.1,2.3,2.6,3.0,3.4,3.7]
    mean_fid = np.zeros(len(Trange))
    qEA = np.zeros(len(Trange))
    fid = {}

    for n_flip in [4]:
        for i, T in enumerate(Trange):
            data, _ = load_data_par(model,T=T,n_step=n_step, n_flip=n_flip)
            n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
            protocol = np.vstack(protocol)
            afid_sort = np.argsort(fid)
            modelpca = PCA(n_components=2)
            #xtsne= TSNE(n_components=2).fit_transform(protocol[afid_sort[-1000:]])
            xpca = modelpca.fit_transform(protocol[:1000])
            #a=protocol[afid_sort[-1000:]]
            #print(a.shape)
            #exit()
            #xtsne = modeltsne.fit_transform(protocol[afid_sort[-1000:]])
            #xpca = modelpca.fit_transform(protocol[afid_sort[-1000:])
            plt.scatter(xpca[:,0],xpca[:,1], label="T=%.2f"%T)
            #plt.scatter(xpca[:,0],xpca[:,1], label="T=%.2f"%T)
            #plt.xlabel("TSNE-1")
            #plt.ylabel("TSNE-2")
            plt.xlabel("PCA-1=%.2f"%(modelpca.explained_variance_ratio_[0]))
            plt.ylabel("PCA-2=%.2f"%(modelpca.explained_variance_ratio_[1]))
            plt.legend(loc='best', fontsize=fontsize)
            plt.tight_layout(pad=0.2)
            plt.savefig('plots/PCA_L=%i_nStep=%i_T=%.2f_nflip=%i.pdf'%(L,n_step,T,n_flip))
            if show is True:
                plt.show()
            else:
                plt.clf()
def complexity():
    """ complexity plot here """



def mean_fidelity(L=1, n_flip=1,fontsize=16, show=True, maxfid='None'):
    argv = sys.argv
    argv.append("L=%i"%L)
    model = QSP(argv = argv)

    """ n_eval, fid, energy, n_visit, protocol """

    mean_fid = np.zeros(40)
    max_fid = np.zeros(40)
    #std_fid = np.zeros(40)
    qEA = np.zeros(40)
    Trange = np.arange(0.1,4.01,0.1)
    n_step = 40

    #for n_step in [20,24,28,32,36,40]:
    for n_flip in [1,2,3,4]:
        for i, T in enumerate(Trange):
            data, _ = load_data_par(model,T=T,n_step=n_step, n_flip=n_flip)
            n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
            #qEA[i] = qfunction(protocol)
            mean_fid[i] = np.mean(fid)
            max_fid[i] = np.max(fid)

        #plt.scatter(Trange,qEA,c="k")
        if maxfid is not None:
            plt.plot(Trange, np.abs(np.log(1-max_fid)), label="SD%i"%n_flip, marker='o', lw=1)
        else:
            plt.plot(Trange, np.abs(np.log(1-mean_fid)), label="SD%i"%n_flip, marker='o', lw=1)

    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('$T$')
    plt.ylabel('$|\log(1-F_h)|$')
    plt.tight_layout(w_pad=0.4)

    if maxfid is not None:
        plt.savefig('plots/max_fid_L=%i_n_step=40.pdf'%(L))
    else:
        plt.savefig('plots/mean_fid_L=%i_n_step=40.pdf'%(L))
    if show is True:
        plt.show()
    plt.clf()



def q_function(L=1, n_flip=1,fontsize=16, show=True):

    argv = sys.argv
    argv.append("L=%i"%L)
    model = QSP(argv = argv)

    """ n_eval, fid, energy, n_visit, protocol """

    mean_fid = np.zeros(40)
    qEA = np.zeros(40)
    Trange = np.arange(0.1,4.01,0.1)

    for n_step in [20,24,28,32,36,40]:
        for i, T in enumerate(np.arange(0.1,4.01,0.1)):
            data, _ = load_data_par(model,T=T,n_step=n_step, n_flip=n_flip)
            n_eval, fid, energy, n_visit, protocol = [[d[i] for d in data] for i in range(5)]
            qEA[i] = qfunction(protocol)

        #plt.scatter(Trange,qEA,c="k")
        plt.plot(Trange, qEA,label="$N_T=%i$"%n_step,marker='o', lw=1)

    plt.legend(loc='best', fontsize=fontsize)
    plt.xlabel('$T$')
    plt.ylabel('$q(T)$')
    plt.tight_layout(pad=0.2)
    plt.savefig('plots/q_L=%i_n_flip=%i.pdf'%(L,n_flip))
    if show is True:
        plt.show()
    plt.clf()

def qfunction(protocol_array):
    return 4.*np.mean(np.var(protocol_array,axis=0))

def mean_excitation(model, protocol):
    n_step = len(protocol[0])
    n_protocol = len(protocol)
    exc = []

    for i, p in enumerate(protocol[:100]):
        #print(i)
        Ei = model.evaluate_protocol(p)
        tmp = np.copy(p)
        for i in range(n_step):
            tmp[i]^=1
            dE = np.log(model.evaluate_protocol(tmp))-np.log(Ei)
            tmp[i]^=1
            exc.append(dE)

    return np.mean(exc), np.std(exc)

if __name__ == "__main__":
    main()