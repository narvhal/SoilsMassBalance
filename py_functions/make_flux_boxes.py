
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from io import BytesIO

def fltr(df, scenario, selcol, selcolval, sample_id = None):
    dft = df[(df.default_scenario == scenario) & (df.select_col == selcol) & (df.select_col_val == selcolval)  ].copy()
    if sample_id != None:
        dft = dft[dft.sample_id.isin(sample_id)].copy()
    return dft
def fltr2(df,  selcol, selcolval, sample_id = None):
    # dft = df[(df.select_col == selcol) & (df.select_col_val == selcolval)  ].copy()
    # if sample_id != None:
    dft = df[df.sample_id.isin(sample_id)].copy()
    return dft

def vcols(listofcols):
    return [c + '_val' for c in listofcols]

def findcols(df, schstr, notstr = None):
    lst = df.columns.to_list()
    if notstr != None:
        for ns in notstr:
            lst = [c for c in lst if not (ns in c)]
    return [c for c in lst if schstr in c]



def wrap_flux_box_visual(df, saveloc , scenario ,selcol,selcolval):
    saveloc2 = saveloc + '\\flux_box_figs_'+ scenario + '_'+selcol+ '_'+ str(selcolval)
    if not os.path.exists(saveloc2):
        os.mkdir(saveloc2)
    for si in ['MT120', 'MT130', 'MT140',  'NQT0', 'NQCC2']:
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])
        list_of_tuplelists,ft,fst = make_into_area(dft, flag_model = 'not', height = 3)

        plot_patches(list_of_tuplelists, dft, ft,fst, height = 3)
        filenametag = si +'_boxflux_w_dust'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')
#         plt.close('all')

    for si in ['MT120', 'MT130', 'MT140',  'NQT0', 'NQCV2']:
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])

        list_of_tuplelists,ft,fst = make_into_area(dft, flag_model = 'simple', height = 3)

        plot_patches(list_of_tuplelists, dft, ft,fst, height = 3)
        filenametag = si +'_boxflux_simple'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')

def proc(key):
    st.info(st.session_state[key])


def mtfmt(mt):   # functions to provide vals for 'model_type'
    mtd = {'simple':"Solve for dissolved flux, no dust ('simple')",
    'wdust':  "Solve for dust flux (dissolved flux constrained by calcite mass balance)" }
    return mtd[mt]

def wrap_flux_box_streamlit(dft, selval_dict):
    fig, ax = plt.subplots()
    height = selval_dict['model_shape']
    flag_model = selval_dict['model_type']
    list_of_tuplelists,ft,fst, height = make_into_area_streamlit(dft, flag_model = flag_model, height = height)
    maxynot, eqlocx = plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False,flag_annot = False)
    fig.set_size_inches(selval_dict['figwidth'], selval_dict['figheight'])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width = 600)
    return fig


def make_into_area_streamlit(df, flag_model= 'simple', height = 'auto' ):
    if flag_model == 'simple':
        fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,  'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
        ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
        spacerloc = 0
    else:
        fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
             'F_coarse_g_m2_yr' ,
             'F_fines_from_br_g_m2_yr' ,
             'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
        ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
        spacerloc = 1
    shape_buffer = .5
    if height == 'Uniform height':
        Fb_L = (df[fmcols[0]].to_numpy()[0] )**(0.5)  # bedrock length where Fb_L**2 = Fb
        height = Fb_L
    elif isinstance(height, float):
        Fb_L = df[fmcols[0]].to_numpy()[0]/height
    else:  # squares
        Fb_L = df[fmcols[0]].to_numpy()[0]**(0.5)
        height = Fb_L

    H = [height]
    L = [Fb_L]
    csum = 0
    XY = [0]
    fst = [df[fmcols[0]].to_numpy()[0]]
    for i, col in enumerate(fmcols[1:]):
        # Actual dimensions of each box:
        # st.write(col)
        # st.write(col in df.columns.to_list())
        colval = df[col].to_numpy()[0]

        if isinstance(colval, str):
            numstr = ufloat_fromstr(colval)
            L1 = numstr.nominal_value/Fb_L
        else:
            if colval>0:
                L1 = colval/height
            else:
                L1 = 0
        if i == spacerloc:
            csum = csum + shape_buffer
        L.append(L1)
        fst.append(colval)
        H.append(height)
#         print('col: ', col, 'colval: ','{:0.2f}'.format(colval),'colval/height: ','{:0.2f}'.format(colval/height),'height: ',height, 'L1: ', '{:0.2f}'.format(L1) )
        # xy coord of Lower Left corner: :
        csum = L[i]+csum +shape_buffer
        XY.append( csum)

    # Now need to make each corners
    list_of_tuplelists= []
    for i, (x,y) in enumerate(zip(L,H)):
        x0 = XY[i]
        x1 = x0 + x
        newxy = [(x0,.1)]
        UL = (x0, y+.1)
#         print(y, H[i])
        DR = (x1, .1)
        UR = (x1, y+.1)
        list_of_tuplelists.append(newxy + [UL] + [UR]+[DR] +newxy)

    # ft = column labels, fst = values of each column, height =
    return list_of_tuplelists, ft, fst, height



def wrap_flux_box_visual2(df, saveloc , scenario ,selcol,selcolval):
    saveloc2 = saveloc + '\\flux_box_figs_'+ scenario + '_'+selcol+ '_'+ str(selcolval)
    if not os.path.exists(saveloc2):
        os.mkdir(saveloc2)
    for si in ['MT120',  'MT140',  'NQT0', 'NQCV2']:

        height = 'auto'
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])
        print(len(dft))
        fig, ax = plt.subplots(nrows = 3, gridspec_kw={'height_ratios': [3, 1, 3]})
        plt.sca(ax[1])
        frame1 = ax[1]
        frame1.axes.get_yaxis().set_visible(False)
        frame1.axis('off')
        # plot_patches(list_of_tuplelists, dft, ft,fst, height = 3, flag_model =flag_model, newfig = False, flag_annot = True)
        plt.sca(ax[2])

        flag_model = 'not'
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        maxynot, eqlocx = plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False,flag_annot = False)

        flag_model = 'simple'

        plt.sca(ax[0])
        dblht = height *2
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        xoffs = eqlocx - list_of_tuplelists[0][3][0] - .5

        plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False, flag_annot = False,set_maxy = maxynot, xoffset = xoffs)

        fig.set_size_inches(10,5)
        filenametag = si +'_boxflux_aligned'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')
        plt.close('all')

def wrap_flux_box_visual3_2(df, saveloc , scenario ,selcol,selcolval):
    saveloc2 = saveloc + '\\flux_box_figs_'+ scenario + '_'+selcol+ '_'+ str(selcolval)
    if not os.path.exists(saveloc2):
        os.mkdir(saveloc2)
    for si in ['MT120',  'MT130',  'NQT0', 'NQCV2']:

        height = 'auto'
        dft =  fltr2(df, selcol, selcolval, sample_id = [si])
        print(len(dft))
        fig, ax = plt.subplots(nrows = 3, gridspec_kw={'height_ratios': [3, 1, 3]})
        plt.sca(ax[1])
        frame1 = ax[1]
        frame1.axes.get_yaxis().set_visible(False)
        frame1.axis('off')
        # plot_patches(list_of_tuplelists, dft, ft,fst, height = 3, flag_model =flag_model, newfig = False, flag_annot = True)
        plt.sca(ax[2])

        flag_model = 'not'
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        maxynot, eqlocx = plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False,flag_annot = False)

        flag_model = 'simple'

        plt.sca(ax[0])
        dblht = height *2
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        xoffs = eqlocx - list_of_tuplelists[0][3][0] - .5

        plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False, flag_annot = False,set_maxy = maxynot, xoffset = xoffs)

        fig.set_size_inches(10,5)
        filenametag = si +'_boxflux_aligned'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')
        plt.close('all')




def wrap_flux_box_visual3(df, saveloc , scenario ,selcol,selcolval):



    # Need to get CaCO3 values for rock and soils (fines) + magnitude of clasts
    # Bedrock -- show carbonate/non carbonate within box ~ 99% carb
    # fine soil == show carb/non carb within box  -- 5-25% carb  (10% NQ, up to 30% MT1)
    # lithic fragments same ~ 99% carb 
    # Also show dust CaCO3? 

    # CaCO3 +MgCO3 conservation (assuming maximum amt in each quantity)



    saveloc2 = saveloc + '\\flux_box_figs_'+ scenario + '_'+selcol+ '_'+ str(selcolval)
    if not os.path.exists(saveloc2):
        os.mkdir(saveloc2)
    for si in ['MT120',  'MT140',  'NQT0', 'NQCV2']:

        height = 'auto'
        dft = fltr(df, scenario, selcol, selcolval, sample_id = [si])
        print(len(dft))
        fig, ax = plt.subplots(nrows = 3, gridspec_kw={'height_ratios': [3, 1, 3]})
        plt.sca(ax[1])
        frame1 = ax[1]
        frame1.axes.get_yaxis().set_visible(False)
        frame1.axis('off')
        # plot_patches(list_of_tuplelists, dft, ft,fst, height = 3, flag_model =flag_model, newfig = False, flag_annot = True)
        plt.sca(ax[2])

        flag_model = 'not'
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        maxynot, eqlocx = plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False,flag_annot = False)

        flag_model = 'simple'

        plt.sca(ax[0])
        dblht = height *2
        list_of_tuplelists,ft,fst, height = make_into_area(dft, flag_model = flag_model, height = height)
        xoffs = eqlocx - list_of_tuplelists[0][3][0] - .5

        plot_patches(list_of_tuplelists, dft, ft,fst, height = height, flag_model =flag_model, newfig = False, flag_annot = False,set_maxy = maxynot, xoffset = xoffs)

        fig.set_size_inches(10,5)
        filenametag = si +'_boxflux_aligned'

        savefig(filenametag,
            saveloc2,
            [],
            [],
            (1,2),
            w_legend=False,
            prefixtag='')
        plt.close('all')




def plot_patches(list_of_tuplelist, df, ft,fst,add_conc = 'auto',  height = 'auto', flag_model = 'simple', newfig = True, flag_annot = True, set_maxy = None, xoffset = 0):
    if newfig:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()
        fig = plt.gcf()
    equals_locx = 0
    if height != 'auto':
        maxy = height
    else:
        maxy = 1
    if flag_model == 'simple':
        hch = ['x', '', '', '']
        bxc = ['grey', 'rosybrown', 'indianred', 'lightcyan']
    else:
        hch = ['x','', '', '', '', '']
        bxc = ['grey','burlywood', 'rosybrown', 'indianred', 'lightcyan', 'burlywood']
    flag_tilt_label = False

    if flag_annot!= True:
        for i, points in enumerate(list_of_tuplelist):
            adjx = [points[p][0] + xoffset for p in np.arange(len(points))]
            y = [points[p][1] for p in np.arange(len(points))]
            npp = list(zip(adjx, y))
            ax.add_patch(mpatches.Polygon(npp, ec = 'dimgrey', fc = bxc[i], hatch = hch[i], ls = '-', lw = .5))  # df.cdict.iloc[0]
            npn = (npp[0][0], (npp[0][1]+npp[1][1])/2 )

            if (points[3][0] - points[0][0])<=.8:
                plt.annotate(' '+ft[i]+'   : {:0.1f}'.format(fst[i]), npp[1], rotation = 45, fontsize = 15)
                if (points[3][0] - points[0][0])>=.6:
                    plt.annotate(''+ft[i], npn, va = 'center')
                flag_tilt_label = True
            else: # LABEL boxes in middle
                plt.annotate(' '+ft[i], npn, va = 'center', fontsize = 15)
                plt.annotate('\n \n \n  {:0.1f}'.format(fst[i]), npn, va = 'center')
            # Add equation stuff to nearby box
            if i>0:
                spacex = (npp[0][0] - (list_of_tuplelist[i-1][3][0] + xoffset))/2
                if flag_model == 'simple':
    #                 print(i, points)
                    syms = [' ', '=', '+', '+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center')
                    if i == 1:
                        equals_locx = npp[1][0]-spacex
                else:
                    syms = [' ','+', '=', '+', '+','+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center')
                    if i == 2:
                        equals_locx = npp[1][0]-spacex

            xy = [maxy]+adjx
            maxy = np.max(xy)

        frame1 = plt.gca()
        if set_maxy !=None:
            plt.xlim(0, set_maxy+0.3)
            plt.ylim(0, height*2) #set_maxy/3+0.1 )

            frame1 = plt.gca()

            plt.annotate(df.sample_id.iloc[0], (0, set_maxy/3+0.1 -1.5)) #(0, npp[1][1]+4.5))
            plt.annotate('(g/m$^2$/yr)', (0, set_maxy/3+0.1 -3.5 ) )#npp[1][1]+3.5))
            plt.annotate(df.sample_region.iloc[0], (0, set_maxy/3+0.1 -2.4) ) #npp[1][1]+1.5))

        else:
            plt.xlim(0, maxy+0.3 )
            plt.ylim(0, height*2+0.1 )
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        x = maxy+.4
        if flag_tilt_label:
            y = maxy/2+1.1
        else:
            y = maxy/2+0.1

        if height == 'auto':
            fig.set_size_inches(x,y)
        else:
            fig.set_size_inches(height*2+5, height)

    elif flag_annot:
        frame1 = plt.gca()

        plt.annotate(df.sample_id.iloc[0], (0, 1)) #(0, npp[1][1]+4.5))
        plt.annotate('(g/m$^2$/yr)', (0, .1) )#npp[1][1]+3.5))
        plt.annotate(df.sample_region.iloc[0], (0, .5) ) #npp[1][1]+1.5))


    frame1.axis('off')
    return maxy, equals_locx
#     hch = ['x', '', '', '']
#     hch = ['x','', '', '', '', '']
#     bxc = ['grey', 'rosybrown', 'indianred', 'lightcyan']
#     bxc = ['grey','burlywood', 'rosybrown', 'indianred', 'lightcyan', 'burlywood']
