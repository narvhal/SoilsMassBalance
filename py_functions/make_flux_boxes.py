
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
    list_of_tuplelists,ft,fst, height , L, H, XY, fst, YC = make_into_area_streamlit(dft, flag_model = flag_model, height = height, scale = selval_dict["boxscale"] , shape_buffer = selval_dict["shape_buffer"])
    maxynot, eqlocx = plot_patches(list_of_tuplelists, dft, ft, L, H, XY,YC, fst, height = height,
        flag_model =flag_model, newfig = False,flag_annot = False)
    fig.set_size_inches(selval_dict['figwidth'], selval_dict['figheight'])
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf, width = selval_dict["pixelwidth"])
    return fig


def deal_w_ustring(val):
    if isinstance(val, str):
        numstr = ufloat_fromstr(val)
    else:
        numstr = val
    return numstr

def make_into_area_streamlit(df, flag_model= 'simple', height = 'auto', scale = 1, shape_buffer = .75):
    if flag_model == 'simple':
        fmcols = vcols([ 'F_br_g_m2_yr' , 'F_coarse_g_m2_yr' ,  'F_fines_boxmodel_g_m2_yr' ,
            'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr'  ])
        ft = ['F$_b$', 'F$_c$', 'F$_f$', 'F$_{dis}$']
        spacerloc = 0
    else:
        fmcols = vcols([ 'F_br_g_m2_yr' ,'F_dust_g_m2_yr',
             'F_coarse_g_m2_yr' ,
             'F_fines_from_br_g_m2_yr' ,
             'F_dissolved_g_m2_yr','F_dust_g_m2_yr' ])
        ft = ['F$_b$','F$_{dust}$', 'F$_c$', 'F$_{f,br}$', 'F$_{dis}$', 'F$_{dust}$']
        spacerloc = 1


    H = []  # Vertical
    L = []  # Horiz
    csum = shape_buffer
    fst = []
    Fbr_L = df[fmcols[0]].to_numpy()[0]   # BR value
    XC = [csum]


    for i, col in enumerate(fmcols[0:]):
        colval = df[col].to_numpy()[0]
        colval = deal_w_ustring(colval)
        if colval > 0:
            if height == 'Uniform height':  # About half of sqrt (FBr_L)
                htt = 0.3 * (Fbr_L)**(0.5)  # bedrock length where Fb_L**2 = Fb
                L1 = colval/htt*scale
            elif isinstance(height, float):
                htt = height
                L1 = colval/htt*scale
            else:  # squares
                L1 = (colval*scale)**(0.5)
                htt = L1
        else:
            L1 = 0

        if i == spacerloc:
            csum = csum + shape_buffer
        # st.write("Make area L1xH: {:.1f}x {:.1f} = {:.1f}".format(float(L1), float(htt), float(L1)*float(htt)))
        # st.write(" {:s}   Orig Area: {:.1f}".format(str(np.round(colval, 1) ==np.round(float(L1)*float(htt), 1) ), colval))
        L.append(L1)
        fst.append(colval)
        H.append(htt)
        csum = L1+csum +shape_buffer
        XC.append( csum)
    # Now need to make each corners
    list_of_tuplelists= []
    for i, (x,y) in enumerate(zip(L,H)):
        # y0 is 0.1 basically
        YC = []
        flag_along_baseline = False
        if flag_along_baseline:
            x0 = XC[i]
            x1 = x0 + x
            y0 = 0.1
            y1 = y0+y
        else: # along centerline
            midy = np.max(H)/2
            x0 = XC[i]
            x1 =  x0 + x
            y0 = midy-(y/2)
            y1 = midy+(y/2)
        YC.append((y0, y1))
        DL = (x0,y0)
        UL = (x0, y1)
        DR = (x1, y0)
        UR = (x1, y1)
        # st.write("Make area from points: {:.1f}x {:.1f} = {:.1f}".format(float(x1-x0), float(y1-y0), float((x1-x0) *(y1-y0))))
        # st.write("Make area from x y : {:.1f}x {:.1f} = {:.1f}".format(float(x1-x0), float(y1-y0),float(x*y)))
        # colval = df[fmcols[i]].to_numpy()[0]
        # st.write(" {:s}   Orig Area: {:.1f}".format(str(np.round(colval, 1) ==np.round(x*y, 1) ), colval))

        list_of_tuplelists.append([DL] + [UL] + [UR]+[DR] +[DL])
    return list_of_tuplelists, ft, fst, height, L, H, XC, fst, YC


def plot_patches(list_of_tuplelist, df, ft, L, H, XC, YC, fst,add_conc = 'auto',  height = 'auto', flag_model = 'simple',
    newfig = True, flag_annot = True, set_maxy = None, xoffset = 0):
    if newfig:
        fig, ax = plt.subplots()
    else:
        ax = plt.gca()
        fig = plt.gcf()
    equals_locx = 0

    if height != 'Uniform height':
        maxy = H[0]   # bedrock height
        maxx = XC[-1]
    elif isinstance(height, float):
        maxy = height
        maxx = XC[-1]
    else:# squares
        maxy = np.max(H)
        maxx = XC[-1]
    # st.write("XY",XY)
    mxo = maxx
    if flag_model == 'simple':
        hch = ['x', '', '', '']
        bxc = ['grey', 'rosybrown', 'indianred', 'lightcyan']
    else:
        hch = ['x','', '', '', '', '']
        bxc = ['grey','burlywood', 'rosybrown', 'indianred', 'lightcyan', 'burlywood']
    flag_tilt_label = False

    mxx = []
    for i, points in enumerate(list_of_tuplelist):
        flag_along_baseline = False
        if flag_along_baseline: pass
        else:
            midy = np.max(H)/2
            adjx = [points[p][0] + xoffset for p in np.arange(len(points))]
            y = [points[p][1] for p in np.arange(len(points))]
            npp = list(zip(adjx, y))
            ax.add_patch(mpatches.Polygon(npp, ec = 'dimgrey', fc = bxc[i], hatch = hch[i], ls = '-', lw = .5))  # df.cdict.iloc[0]
            # npn = ( (npp[0][1]+npp[1][1])/2 ,  (npp[1][0] + npp[0][0])/2 )  # Find x and y-midpoint
            npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2, midy )  # Find x and y-midpoint
            # st.write(npp)
            # st.write(f"Area {ft[i]}: {(npp[1][1]-npp[1][0])* (npp[2][0]-npp[1][0])}")
            # st.write(f"Orig: {fst[i]}" )
            # st.write("npp Points:",npp[1])
            if flag_model == 'simple':
                pct_denom = fst[0]  # just bedrock flux
            else:
                pct_denom = fst[0] + fst[1] # bedrock + dust flux
            fst_as_pct = np.round(fst[i]/pct_denom*100, 0)
            if flag_annot:
                if (points[3][0] - points[0][0])<=maxx/20:
                    # st.write("narrow box, "+ft[i]+'   : {:0.1f}'.format(fst[i]))
                    plt.annotate(' '+ft[i]+' : {:0.1f}'.format(fst[i]), npp[1], rotation = 45, fontsize = 15)
                    # if (points[3][0] - points[0][0])>=.6:
                        # plt.annotate(''+ft[i], npn, va = 'center')
                    flag_tilt_label = True
                else: # LABEL boxes in middle
                    # st.write("wide box, " +ft[i])
                    # st.write(npn)
                    # st.write(npp[0])
                    # st.write(points[0])
                    # st.write()

                    plt.annotate(ft[i], npn, va = 'center', fontsize = 15, ha = 'center')
                    plt.annotate('\n{:0.1f}'.format(fst[i]), npn, va = 'top', ha = 'center')
                    plt.annotate('\n\n {:0.0f}%'.format(fst_as_pct), npn, va = 'top', ha = 'center')
            else:
                # npn = (npn[0], npn[1]+ midy*1.9)
                npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  npn[1]+ midy*1.9 ) # Find x and y-midpoint

                # npn = (npp[0][0] , npn[1]+ midy*1.9)
                plt.annotate(ft[i], npn, va = 'center', fontsize = 15, ha = 'center')
                if i == 0:
                    plt.annotate('\n {:0.1f}\n g/m$^2$/yr'.format(fst[i]), npn, fontsize = 10,  va = 'top', ha = 'center')
                else:
                    plt.annotate('\n {:0.1f}'.format(fst[i]), npn,fontsize = 10, va = 'top', ha = 'center')

                npn = (npp[0][0] + (npp[3][0]-npp[0][0])/2,  (npp[0][1]+npp[1][1])/2 ) # Find x and y-midpoint
                plt.annotate('{:0.0f}%'.format(fst_as_pct), npn, va = 'center', ha = 'center', fontsize = 8, fontweight = "bold")
            # plt.annotate(f"LxH = Area\n{L[i]} x {H[i]} \n\t= {fst[i]}", (points[0][0], 0.1), va = "center", rotation = 20)
            # Add equation stuff to nearby box
            if i>0:
                spacex = (npp[0][0] - (list_of_tuplelist[i-1][3][0] + xoffset))/2
                if flag_model == 'simple':
    #                 print(i, points)
                    syms = [' ', '=', '+', '+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center', va = 'center')

                    if i == 1:
                        equals_locx = npp[1][0]-spacex
                else:
                    syms = [' ','+', '=', '+', '+','+', ' ']
                    sy = syms[i]
                    plt.annotate(sy, (npp[1][0]-spacex, (npp[0][1]+npp[1][1])/2 ),ha='center', va = 'center')

                    if i == 2:
                        equals_locx = npp[1][0]-spacex
        mxx.append(adjx)
    maxx2 = np.max(np.array(mxx))
    frame1 = plt.gca()
    if set_maxy !=None:
        maxx = set_maxy
        plt.xlim(0, set_maxy+0.3)
        plt.ylim(0, maxy*2) #set_maxy/3+0.1 )

        frame1 = plt.gca()
    else:
        # st.write("TWO max x",maxx)
        # st.write("TWO max Y",maxy)
        xl = maxx2*1.1+0.3
        yl = maxy*2+0.1
        # st.write(f"xy lims: {xl}, {yl}" )
        plt.xlim(0, xl)
        plt.ylim(0, yl )
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

    plt.annotate(df.sample_id.iloc[0] +"\n" + df.sample_region.iloc[0], (0, maxy+ 13/14*maxy), fontsize = 13) #(0, npp[1][1]+4.5))
    #npp[1][1]+3.5))

    frame1.axis('off')
    return maxy, equals_locx
