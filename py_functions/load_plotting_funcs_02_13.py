
def wrap_newdf2(dfto,K = 14.6, print_df_short = False, summary_val_name = 'none',  col_overwrite = 'none', Dcol_overwrite = 'none', dcol_dict_val = [], vals_arr = 0,dff_overwrite = 'none', dff_val = [], flag_br_equiv = False, flag_coarse_subsurface = False, flag_pde = False, tn = 'TableName',saveloc = 'F:\Ch4_Soils\Tables', dirn = '\\24_26_01', dict_of_fmt = None):
    if summary_val_name != 'none':
        col = summary_val_name
        # print('col is summary_val_name')
    else:
        col = col_overwrite[0]

    dff = write_defaults_to_df2(dfto,col_overwrite= dff_overwrite, dict_val_overwrite =dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val)
    #     print('Default df cols: ', dff.columns.to_list())
    #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
    #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
    if isinstance(dff_overwrite, list):  # Then we wanna calculate non-default vals
        for i, ovrc in enumerate(dff_overwrite):
            ovrval = dff_val[i] # Will write uniform value for all samples
            if ovrc not in dff.columns.to_list():
                print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
            dff[ovrc] = ovrval

    dff.drop(['sample_region', 'sample_id'], axis = 1, inplace = True)

    for i, val in enumerate(vals_arr):
        if flag_coarse_subsurface:
            dft = get_new_df_results_w_unc2(dfto,dff, col_overwrite = col_overwrite, val_overwrite = [val], flag_br_equiv = flag_br_equiv, flag_coarse_subsurface = val, flag_pde = flag_pde)
        else:
            dft = get_new_df_results_w_unc2(dfto,dff, col_overwrite = col_overwrite, val_overwrite = [val], flag_br_equiv = flag_br_equiv, flag_coarse_subsurface = flag_coarse_subsurface, flag_pde = flag_pde)
        dft['select_col'] = col
        dft['select_col_val'] = val
        if i == 0:
            dfn = dft.copy()
        else:
            dfn = pd.concat([dfn, dft])

    # calculate expected curvature IF dust flux and bedrock erosion rates
    # Ben asher equation.
    #     # Units:
    # F_br = dfn['F_br_val']
    # F_dust = dfn['F_dust_val']
    # DF = dfn['DF_val']
    # p_re = dfn['p_re_val']
    # dfn['K'] = dfn.sample_region.map({'AZ':2.6, 'SP':14.6})
    # dfn['ht_cv_actual_topo'] = dfn.sample_id.map({'MT120': 0.058, 'NQT0': 0.018, 'NQCV2': 0.018})  # from Ch 3 table 1

    # dfn['CDF'] = (1/DF * (F_br - dfn['F_coarse']) + 1*dfn['F_coarse'])/F_br  ## needs to incorporate coarse sediment as well

    # ht_cv = -(F_br + F_dust)*(1-dfn['CDF'] )/p_re / K
    # dfn['ht_cv_BenAsher'] = ht_cv

    pc = ['sample_id','select_col', 'select_col_val', 'D_val', 'z_val', 'F_coarse_g_m2_yr_val','F_br_g_m2_yr',   'F_fines_boxmodel_g_m2_yr', 'F_dust_g_m2_yr', 'dust_10Be_conc_diff_w_gralySP', 'dust_10Be_conc_diff_w_gralyAZ']
    if print_df_short:
        dfpresamp =  dfn[dfn.sample_id.isin(['MT120', 'NQT0', 'NQCV2'])].copy()
    #         print(dfpresamp[pc].sort_values(by=['sample_id', 'F_dust_g_m2_yr']))
    if not os.path.exists(saveloc):
        os.mkdir(saveloc)
    if not os.path.exists(saveloc +  dirn):
        os.mkdir(saveloc+ dirn)
    # also need to write xlsx
    sn =  saveloc+dirn +'\\' + tn
    write_to_excel(dfn, sn)

    # Also need script to write Latex txt
    latex_col_order = pc
    # reduce rows for Latex and plots -- only samples we care about. not polop
    dfn =select_df_rows(dfn)

    write_just_latex(dfn, tn, saveloc+dirn,latex_col_order , dict_of_fmt = dict_of_fmt)
    sn = saveloc+dirn +'\\' + tn  + '.txt'
    edit_latex_tbl(sn)

    # also need to make + save plots.
    dfn = map_cdict(dfn)
    #     print(dfn.columns.to_list())

    # colors are now columns
    return dfn

def set_up_df_for_flux_results3(df,dff, Ncol = 'calc-10Be_at_g', N_unc_col = 'calc-uncert_10Be_at-g', I_desc = 'site'):
    # Apply dflt values to df for each row (sample id)
    # Flat values for all rows

    dd = {'N_col': Ncol,  'N_unc_col': N_unc_col, 'Inv_Desc': I_desc}
    #, 'p_re': p_re, 'p_re_unc': p_re_unc,'p_br': p_br,  'D_Desc': D_Desc, 'D': D, 'D_unc': D_unc, 'DF': DF, 'DF_unc': DF_unc, 'br_E_rate':br_E_rate, 'br_E_rate_unc': br_E_rate_unc, 'coarse_mass':coarse_mass, 'coarse_mass_unc': coarse_mass_unc, 'coarse_area': coarse_area, 'coarse_area_unc': coarse_area_unc, 'max_coarse_residence_time': max_coarse_residence_time, 'max_coarse_residence_time_unc': max_coarse_residence_time_unc }
    zcol = dff['zcol'].iloc[0]
    if zcol in ['local_sd_min', 'local_sd_max', 'local_sd_avg']:
        dft = df[['sample_region', 'sample_location', 'sample_position','sample_id','s_dx (cm)','s_dx (cm) upslope','inv_sum_flag','flow acc m2', Ncol,N_unc_col, zcol]].copy()

    else:
        dft = df[['sample_region', 'sample_location', 'sample_position','sample_id','s_dx (cm)','s_dx (cm) upslope','inv_sum_flag','flow acc m2', Ncol,N_unc_col]].copy()

    for i, di in enumerate(dd.keys()):
        dft[di] = dd[di]

    dft = dft.join(dff, on='sample_id', how='left')

    if zcol in ['local_sd_min', 'local_sd_max', 'local_sd_avg']:
        SD = dft[zcol]
    else:
        SD = zcol
    dft['z'] = SD
    if 'z_unc' not in dff.columns.to_list(): z_unc = 0
    else: z_unc = dff['z_unc'].iloc[0]
    dft['z_unc'] = z_unc

    if I_desc == 'site':
        sitecols = []
        dft['z_site'] = dft.groupby('inv_sum_flag')['z'].transform('sum')
        dft['N_site'] = dft.groupby('inv_sum_flag')[Ncol].transform('mean')
        dft['N_site_unc'] = dft.groupby('inv_sum_flag')[N_unc_col].transform('mean')
        dft['N'] = dft.apply(lambda x: ufloat(x['N_site'], x['N_site_unc'], "N"), axis = 1)
        N = dft['N']
        dft['N_unc'] = dft['N_site_unc']

        dft['z'] = dft['z_site']

    else:
        dft['N'] = dft.apply(lambda x: ufloat(x[Ncol], x[N_unc_col], "N"), axis = 1)
        N = dft['N']

    zl = []
    for r, ro in enumerate(dft['z']):
        zl.append(ufloat(ro, dft['z_unc'].iloc[r]))
    dft['z'] = zl

    return dft, SD, N

def get_new_df_results_w_unc2(df,dff, col_overwrite = False, val_overwrite = 0, flag_br_equiv = False, flag_coarse_subsurface = 10, flag_pde = False):
    # col_overwrite = ['coarse_mass'], val_overwrite= 2000,

    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)

    #     print('Default df cols: ', dff.columns.to_list())
    #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
    #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
    if isinstance(col_overwrite, list):  # Then we wanna calculate non-default vals
        for i, ovrc in enumerate(col_overwrite):
            ovrval = val_overwrite[i] # Will write uniform value for all samples
            if ovrc not in dff.columns.to_list():
                print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
            dff[ovrc] = ovrval

    dft, SD, N = set_up_df_for_flux_results3(df,dff)  # does not output ufloats yet
    #     print('L13: debug: ', dft.columns.to_list())
    #     print('L14: dft[\'z\'].iloc[0]', dft['z'].iloc[0])
    #     print('L14: dft[\'coarse_mass\'].iloc[0]', dft['coarse_mass'].iloc[0])
    #     print('L14: dft[\'coarse_area\'].iloc[0]', dft['coarse_area'].iloc[0])

    ## run through and modify anything that isn't ufloat, need to follow uncertainties....
    # Is ufcols just the cols of dff?
    ufcols = ['p_re','p_br', 'br_E_rate', 'coarse_mass', 'coarse_area', 'max_coarse_residence_time', 'z', 'D', 'DF']
    for i, ucol in enumerate(ufcols):#isinstance(uf, uncertainties.core.Variable) | isinstance(uf, uncertainties.core.AffineScalarFunc)\
    #         if ucol == 'coarse_mass':
    #             print('before coarsemass: ', dft[ucol].iloc[0])
        if isinstance(dft[ucol].iloc[0], uncertainties.core.Variable): # already done, don't do anything
    #             print('TRUE: isinstance(dft[ucol].iloc[0], uncertainties.core.Variable)')
            pass
        elif isinstance(dft[ucol].iloc[0],  uncertainties.core.AffineScalarFunc):
    #                print('TRUE: isinstance(dft[ucol].iloc[0],  uncertainties.core.AffineScalarFunc)')
            dft[ucol] = redef_uf(dft[ucol])
        else: # if any of those ufcols isn't already ufloat, then we need to change it and assume the unc col is there else 0
    #             print('TRUE: else, i.e. not a affine scalar func or variable')
            uncol = ucol + '_unc'
            if uncol not in dft.columns.to_list():
    #                 print('uncol: ', uncol)
                dft[uncol] = 0
            tarr = []
            for j, uval in enumerate(dft[ucol]): # redef uf doesn't need to loop through each row but ufloat does
                 tarr.append(ufloat(uval, dft[uncol].iloc[j], ucol))
            dft[ucol] = tarr
    #         if ucol == 'coarse_mass':
    #             print('after ufloate coarsemass: ', dft[ucol].iloc[0])

   # Post-df formation calculations. E.g. coarse mass in subsurface? need to redefine soil depth, bc soil depth is actually FINE soil depth....
    if flag_coarse_subsurface != False:
        if isinstance(flag_coarse_subsurface, float) | isinstance(flag_coarse_subsurface, int):  #flag_coarse_subsurface = percent_subsurface_by_vol_is_coarse
            # print(flag_coarse_subsurface, 'flag coarse subsurface is a float or int')
            SD, coarse_mass = modify_start_subsoil_coarse_seds(dft, flag_coarse_subsurface)
    #         print('get_new_df_results_w_unc:  coarse_mass: ', coarse_mass, '\nSD: ', SD)
            zl = []
            dft['z_old'] = dft['z'].copy()
            dft['coarse_mass_old'] = dft['coarse_mass'].copy()
    #         print(dft['coarse_mass_old'])
            for j, vz in enumerate(dft['z']):
                zl.append(redef_uf(vz - vz*flag_coarse_subsurface/100))
            dft['z'] =zl
            dft['coarse_mass'] = coarse_mass

    dft['Inv'] = dft.apply(lambda x: f_Inv(x['N'],x['p_re'], x['z']), axis = 1)
    dft['Inv_pdeN'] = dft.apply(lambda x: x['Inv'].derivatives[x['N']], axis = 1)
    dft['Inv_pdepre'] = dft.apply(lambda x: x['Inv'].derivatives[x['p_re']], axis = 1)
    pdez = []
    for pd, pdz in enumerate(dft['Inv']):
        zt = dft['z'].iloc[pd]
        pdez.append(pdz.derivatives[zt])
    dft['Inv_pdez'] = pdez
    Inv = dft['Inv']

    dft,res_t = solve_rt(dft)
    dft, E_fines = solve_E_fines(dft)
    # in mm/kyr
    # need mass fluxes --> unc sensitive to res time

    dft, F_fines = solve_F_fines(dft)
    dft, F_coarse  = solve_F_coarse(dft)
    dft, F_br  = solve_F_br(dft)
    dft, F_dust  = solve_F_dust(dft)

    # min conc  at/cm2/yr / g/cm2/yr ==> at/g
    dft['dust_10Be_conc_diff_w_gralySP'] =(dft['D'] - SP_D_graly)/dft['F_dust']
    dft['dust_10Be_conc_diff_w_gralyAZ'] =(dft['D'] - AZ_D_graly)/dft['F_dust']
    #max conc
    dft['dust_10Be_conc_diff_w_gralySP_max'] =( dft['D'] )/dft['F_dust']
    dft['dust_10Be_conc_diff_w_gralyAZ_max'] =(dft['D'] )/dft['F_dust']

    xpos = dft['s_dx (cm)']  # total dist from ridgetop
    x0 = dft['s_dx (cm) upslope']  # dist from ridgetop to upslope sample.
    #     print('reminder: qs is dx from upslope sample')
    #     print('line 83: ', D, xpos, x0, p_re, N)
    Frft = []
    D = dft['D']
    p_re = dft['p_re']
    for f, frf in enumerate(D):
        Frft.append(f_regflux(D[f], xpos[f], x0[f], p_re[f], N[f] ) )  # cm2/yr
    dft['qs_dx_from_upslope_sample']  = Frft

    xpos = dft['flow acc m2']*(100*100)    # m2 --> cm2
    x0 =  dft['s_dx (cm) upslope'] *0

    Frft = []
    D = dft['D']
    for f, frf in enumerate(D):
        Frft.append(f_regflux(D[f], xpos[f], x0[f], p_re[f], N[f] ) )  # cm2/yr
    dft['qs_dx_is_flow_acc_m2'] =   Frft

    # dft['F_br_solids_after_chem_wx'] = (F_br- F_coarse)/dft['DF']

    dft['F_fines_from_br'] = dft['F_fines_boxmodel'] - dft['F_dust']
    dft['F_dissolved'] = (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']

    # These should be equivalent: LHS = RHS of mass balance
    dft['F_br_plus_F_dust'] = dft['F_br'] + dft['F_dust']
    dft['F_coarse_plus_F_fines_plus_F_dissolved']= dft['F_coarse'] + dft['F_fines_boxmodel'] + dft['F_dissolved']

    DF = dft['DF']
    p_re = dft['p_re']


# F:\Ch4_Soils\hilltop_cv_recalc
    dict_nh_6 = {'MT120': -0.038960774739689605, 'MT130': -0.014048258463692646, 'MT140': -0.0039367675781761475, 'MT150': -0.02202351888028153, 'MT160': -0.002990722656353995, 'MT170': 0.0003051757813126851, 'NQT0': -0.025085449218782006, 'NQCC2': 0.019994099934862595, 'NQCV2': 0.01474761962887685, 'NQCV4': 0.04274241129554823, 'NQPR0': 0.1418329874674209}
    dict_nh_10 ={'MT120': -0.03911655970982906, 'MT130': -0.012145996093766386, 'MT140': -0.0047119140625133905, 'MT150': -0.015209089006695816, 'MT160': -0.004703194754474845, 'MT170': -0.004143415178590641, 'NQT0': -0.01717158726283599, 'NQCC2': -0.0025072370256786973, 'NQCV2': -0.00048522949219118776, 'NQCV4': -0.007194301060272058, 'NQPR0': 0.06417105538504028}
    dict_nh_18={'MT120': -0.03511895073784176, 'MT130': -0.012553271449137033, 'MT140': -0.004107586749181472, 'MT150': -0.010360974655399313, 'MT160': -0.00732709582806795, 'MT170': -0.006098664225488383, 'NQT0': -0.0171899561586119, 'NQCC2': -0.005148413778781948, 'NQCV2': -0.008519760285962767, 'NQCV4': -0.006003663768096233, 'NQPR0': 0.0159514345992028}
    dict_nh_26={'MT120': -0.03135276963871658, 'MT130': -0.01397734158119797, 'MT140': -0.004119676053365337, 'MT150': -0.007644201876851472, 'MT160': -0.007458052389321181, 'MT170': -0.006491658506760771, 'NQT0': -0.015380514636356259, 'NQCC2': -0.0003897803903206735, 'NQCV2': -0.012826356570488513, 'NQCV4': -0.0012080726749591534, 'NQPR0': 0.008571178882152648}
    dict_nh_30={'MT120': -0.029847367789513832, 'MT130': -0.014693424460757123, 'MT140': -0.00437018984840149, 'MT150': -0.0072553227364439395, 'MT160': -0.0073637722535687036, 'MT170': -0.006574322092870191, 'NQT0': -0.014767275964152931, 'NQCC2': -0.0004559564374692479, 'NQCV2': -0.01396707100386399, 'NQCV4': 0.00022294262929375627, 'NQPR0': 0.009526058312917179}

    dft['ht_cv_actual_topo_nh6m'] = dft.sample_id.map(dict_nh_6)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh10m'] = dft.sample_id.map(dict_nh_10)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh18m'] = dft.sample_id.map(dict_nh_18)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh26m'] = dft.sample_id.map(dict_nh_26)  # from Ch 3 table 1
    dft['ht_cv_actual_topo_nh30m'] = dft.sample_id.map(dict_nh_30)  # from Ch 3 table 1
    dft['ht_cv_actual_topo'] = dft['ht_cv_actual_topo_nh18m'].copy()
    # dft['ht_cv_actual_topo'] = dft.sample_id.map( {'MT120':0.06, 'MT130': 0.0067, 'MT150': 0.0004, 'NQT0':0.02, 'NQCV2':0.0021, 'NQCV4': 0.0026})

    # These are from cv8sm: {'MT120':0.009, 'MT130': 0.0067, 'MT150': 0.0004, 'NQT0':0.0075, 'NQCV2':0.0021, 'NQCV4': 0.0026})    # Slightly diff locations... these are from Ch 3 --- super high compared to profile curvature... from col DV in 10Be Data reduction....{'MT120': 0.058, 'NQT0': 0.018, 'NQCV2': 0.018})  # from Ch 3 table 1

    # Amtof br that goes to dissolved   / total input
    # dft['CDF'] = ( (dft['F_br'] - dft['F_coarse'])*(1-1/dft['DF'])  - dft['F_coarse'])/dft['F_br']  ##  ## Scaling by density ratios,  needs to incorporate coarse sediment as well
    # dft['CDF_dust'] = ( (dft['F_br'] - dft['F_coarse'])*(1-1/dft['DF'])  - dft['F_coarse'] - dft['F_dust'])/(dft['F_br'] + dft['F_dust'])
    # new CDF Formulation
    dft['CDF'] = dft['F_dissolved'] /dft['F_br']
    dft['CDF_dust'] = dft['F_dissolved'] /(dft['F_br'] + dft['F_dust'])


    # IF FLUXES ARE IN g/m2/yr
    #g/m2/yr/cm2*yr*cm3/g = cm/m2 = cm*1m/100cm/m2 = 1/m/100

    ## IF FLUXES ARE IN g/cm2/yr
    # K is in cm2/yr so it's OK that Fluxes are in g/cm2/yr as well // actually need x100
    # g/cm2/yr / (g/cm3) / (cm2/yr) = cm/yr / (cm2/yr) = 1/cm
    # if ht cv should be in m, then 1/cm * 100 cm/1m = 100 / m
    # p_re should incorporate the coarse sediments as well....
    ht_cv = -(dft['F_br'] + dft['F_dust'])*(1-dft['CDF'] )/p_re / dft['K']

    dft['ht_cv_BenAsher'] = ht_cv*100

    dft['ht_cv_nodust'] = -(dft['F_br'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100
    # This one makes no sense: CDF already takes into account the dissolutb dft['ht_cv_dust_minus_dissolution'] = -(dft['F_br']+ dft['F_dust'] - dft['F_dissolved'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100

    dft['ht_cv_noCDF_noDust'] =  -(dft['F_br'] )/p_re / dft['K'] * 100

    dft['ht_cv_dustnoCDF'] = -(dft['F_br']*(1-dft['CDF'] ) +  dft['F_dust'])  /p_re / dft['K'] * 100


    dft['ht_cv_br_dust_nocdf'] =  -(dft['F_br'] + dft['F_dust'])/p_re / dft['K'] * 100

    # See excel sheet --- these simplify the CDF * solid matl as above so probably same
    dft['ht_cv_v1_dust'] = -(dft['F_coarse'] + dft['F_fines_from_br'] + dft['F_dust'] )/ dft['K'] / p_re *100
    dft['ht_cv_v2_brwx'] = -(dft['F_coarse'] + dft['F_fines_from_br'])/ dft['K'] / p_re *100
    dft['ht_cv_v3_nowx_dust'] = -(dft['F_br'] + dft['F_dust'])/ dft['K'] / p_re *100

    ht_cv = -(dft['F_br'] + dft['F_dust'])*(1-dft['CDF_dust'] )/p_re / dft['K']

    dft['ht_cv_BenAsher_CDF_dust'] = ht_cv*100

    dft['ht_cv_nodust_CDF_dust'] = -(dft['F_br'] )*(1-dft['CDF_dust'] )/p_re / dft['K'] * 100
    # This one makes no sense: CDF already takes into account the dissolutb dft['ht_cv_dust_minus_dissolution'] = -(dft['F_br']+ dft['F_dust'] - dft['F_dissolved'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100

    dft['ht_cv_dustnoCDF_CDF_dust'] = -(dft['F_br']*(1-dft['CDF_dust'] ) +  dft['F_dust'])  /p_re / dft['K'] * 100



    # Simple mass balance model:
    # Fbr = Fcoarse + Ffines + Fdissolved, ---> Fbr and Fcoarse and Ffines are the same as in more complexs model, but LHS is smaller so RHS also has to be smaller...
    # RHS simple mass balance model:
    dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']  =dft['F_br']  - dft['F_coarse'] - dft['F_fines_boxmodel']

    #      dft['F_br']+ dft['F_dust'] - dft['F_dissolved']
    #      dft['F_br']+ dft['F_dust']- (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']
    #      dft['F_br']+ dft['F_dust']-  dft['DF'] * dft['F_fines_boxmodel']   + dft['DF'] * dft['F_dust']


    #      dft['F_br'] - dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']
    #      dft['F_br'] -dft['F_br'] +dft['F_coarse'] +dft['F_fines_boxmodel']
     #                             dft['F_coarse'] +dft['F_fines_boxmodel']

    #Also makes no sense:  dft['ht_cv_F_br_minus_simpledissolution'] = -(dft['F_br'] - dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines'])*(1-dft['CDF'] )/p_re / dft['K'] * 100



    # # hilltop differences...
    # for i,nh in enumerate([6,10,18,26,30]):
    #     stnh = str(nh)
    #     dft['ht_cv_actual_minus_nodustcv_nh' + stnh] = dft['ht_cv_actual_topo_nh'+stnh+'m'] - dft['ht_cv_nodust']
    #     dft['ht_cv_actual_minus_dustcv_nh' + stnh] = dft['ht_cv_actual_topo_nh'+stnh+'m'] - dft['ht_cv_BenAsher']
    #     dft['ht_cv_actual_minus_dustnoCDFcv_nh' + stnh] = dft['ht_cv_actual_topo_nh'+stnh+'m'] - dft['ht_cv_dustnoCDF']
    #     dft['ht_cv_actual_minus_noCDFcv_nh' + stnh] = dft['ht_cv_actual_topo_nh'+stnh+'m'] - dft['ht_cv_noCDF_noDust']
    #     # dft['ht_cv_actual_minus_simpledissolutioncv_nh' + stnh] = dft['ht_cv_actual_topo_nh'+stnh+'m'] - dft['ht_cv_F_br_minus_simpledissolution']



    dft['ht_cv_actual_minus_nodustcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_nodust']
    dft['ht_cv_actual_minus_dustcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_BenAsher']
    dft['ht_cv_actual_minus_dustnoCDFcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_dustnoCDF']
    dft['ht_cv_actual_minus_noCDFcv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_noCDF_noDust']
    # dft['ht_cv_actual_minus_simpledissolutioncv'] = dft['ht_cv_actual_topo'] - dft['ht_cv_F_br_minus_simpledissolution']

    dft['ht_cv_diff2_nodustcv_dust_cv'] = dft['ht_cv_actual_minus_nodustcv'] -dft['ht_cv_actual_minus_dustcv']
    dft['ht_cv_diff2_nodustcv_dustnoCDF_cv'] =  dft['ht_cv_actual_minus_nodustcv'] - dft['ht_cv_actual_minus_dustnoCDFcv']
    dft['ht_cv_diff2_dustcv_dustnoCDF_cv'] =  dft['ht_cv_actual_minus_dustcv'] - dft['ht_cv_actual_minus_dustnoCDFcv']

    #  ['ht_cv_diff2_nodustcv_dust_cv', 'ht_cv_diff2_nodustcv_dustnoCDF_cv', 'ht_cv_diff2_dustcv_dustnoCDF_cv']

    # ht_cv_dust_minus_dissolution_val  ht_cv_F_br_minus_simpledissolution_val
    # Simple nodust model : (F_dissolved + F_fines)/F_fines

    dft['Simple_model_apparent_DF'] = (dft['F_br']  - dft['F_coarse'] )/ dft['F_fines_boxmodel']
    dft['Dust_model_apparent_DF'] = (dft['F_br']  - dft['F_coarse'] )/ dft['F_fines_from_br']

    # Can solve for DF if we assume Fd:
    # (F_dust - F_fines) (DF + 1) = F_coarse - F_br
    # F_dust*DF - F_fines*DF + F_dust -F_fines = F_coarse - F_br
    # DF(F_dust - F_fines) + F_dust - F_fines = F_coarse - F_br
    # F_br + F_dust = F_coarse  + F_fines + DF(F_fines-F_dust )
        #DF = (F_coarse - F_br)/(F_dust - F_fines) - 1
    dft['DF_if_Fd_is_set_to_1'] = (F_coarse - F_br)/(1/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_5'] = (F_coarse - F_br)/(5/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_10'] = (F_coarse - F_br)/(10/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_15'] = (F_coarse - F_br)/(15/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_20'] = (F_coarse - F_br)/(20/10000 - F_fines) -1


    # Impt Fluxes as ratios to BR flux
    dft['F_coarse_normbr'] = dft['F_coarse']/dft['F_br']
    dft['F_fines_from_br_normbr'] = dft['F_fines_from_br']/dft['F_br']
    dft['F_fines_boxmodel_normbr'] = dft['F_fines_boxmodel']/dft['F_br']
    dft['F_dust_normbr'] = dft['F_dust']/dft['F_br']
    dft['F_dissolved_normbr'] = dft['F_dissolved']/dft['F_br']
    dft['F_dissolved_simple_normbr'] = dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']/dft['F_br']



    if flag_pde:
        dft = f_Inv_unc_pde(dft, N,p_re,SD, N_unc, p_re_unc, z_unc)
        dft = f_rest_unc_pde(dft, Inv, D, ltl, Inv_unc, D_unc)
        dft = f_erate_unc_pde(dft, SD, D, N, ltl, p_re, z_unc, D_unc, N_unc, p_re_unc)

    to_m2_cols = [co for co in dft.columns.to_list() if co.startswith('F_')]
    # ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust', 'F_br_solids_after_chem_wx','F_fines_from_br', 'F_dissolved', 'F_br_plus_F_dust','F_coarse_plus_F_fines_plus_F_dissolved','F_coarse_normbr', 'F_fines_from_br_normbr','F_fines_boxmodel_normbr', 'F_dust_normbr', 'F_dissolved_normbr', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']
    dft = dft.copy()
    # Change fluxes to m2
    # g/cm2/yr  * 100*100cm2/m2
    for c,cc in enumerate(to_m2_cols):
        dft[cc + '_g_m2_yr'] = dft[cc].apply(lambda x: x*10000).copy()
    dft = dft.copy()

    dft['rt_ky'] = dft['rt'].copy() /1000 # ky
    dft = dft.copy()
    # At very end, run through all columns and make sure none are Affine Scalar Function
    for c, col in enumerate(dft.columns.to_list()):
        fv = dft[col].iloc[0]
        dft = dft.copy()
    # Actually I think remaking all ufloats will improve those horribly long decimals        if isinstance(fv, uncertainties.core.Variable):
    #             dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
        if isinstance(fv, uncertainties.core.AffineScalarFunc) | isinstance(fv, uncertainties.core.Variable):
    #             print('Line 96:' , col, dft[col].iloc[0],'\n', dft[col])
            dft = dft.copy()
            dft[col] = redef_uf(dft[col])

            dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
            dft = dft.copy()

    #             print(col, dft[col].iloc[1])
        else: pass

    return dft


def write_defaults_to_df2(df, col_overwrite= 'none', dict_val_overwrite = [], Dcol_overwrite = 'none', dcol_dict_val = []):
    # Rules:
    # D --> if MT, then Graly AZ, else Graly SP
    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)
    # AZ_D_graly = D_graly(450, 31.2)
    # SP_D_graly = D_graly(450, 39.1)

    if col_overwrite == 'D':
        D_dflt_dct = dict_val_overwrite
    else:
        D_dflt_dct = {'AZ': AZ_D_graly, 'SP': SP_D_graly}

    if Dcol_overwrite != 'none':
        D_dflt_dct = dcol_dict_val

    # DF --> based on cao soil/cao br; SP: ~4% cao in soil so 25 DF, MT: ~ 6% cao in soil so ~ 17 DF. BUT some soils in both samples have ~ 20% cao AND if include MgO then MT 9, 30%, and SP is 7,8, 26%. But I think these are probably sm rock fragments elevating these vals.... so use lowest
    if col_overwrite == 'DF':
        DF_dflt_dct =dict_val_overwrite
    else:
        DF_dflt_dct = {'AZ': 17, 'SP': 25}
    # p_re --> 1.4
    # p_re, p_br, coarse_area, max_coarse_residence_time, p_crs, coarse_mass
    if col_overwrite == 'p_re':
        p_re = dict_val_overwrite
    else:
        p_re = 1.4
    # p_br --> 2.6
    if col_overwrite == 'p_br':
        p_br = dict_val_overwrite
    else:
        p_br = 2.6
    # coarse area
    if col_overwrite == 'coarse_area':
        coarse_area = dict_val_overwrite
    else:
        coarse_area = 300
    # max coarse residence time.
    if col_overwrite == 'coarse_area':
        max_coarse_residence_time = dict_val_overwrite
    else:
        max_coarse_residence_time = 15500
    # density of coarse sediment modify? but if anything I want to overestimate the coarse sedments SHOULD be 2.2 or so (Poesen Laveee 1994)
    p_crs = 2.2
    # coarse mass
    if col_overwrite == 'coarse_mass':
        coarse_mass = dict_val_overwrite
    else:
        coarse_mass = 1133



    if col_overwrite == 'K': # Sed transport coeff
        K_sed = dict_val_overwrite
    else:
    # dft['K'] = dft.sample_region.map()
        K_sed = {'AZ':2.6, 'SP':14.6}
    # # br E rate: use closest E rate to sample....(br or clast), N14 for all meteoric except NQCV4 --> N04 br, NQPR0 --> C01 or BR04-C
        # Br, BRunc, ID, Name  (bulk as measured)
        # 10.2  2.4 13  BH04-B
        # 47.3  8.6 3   Bic-1
        # 24.2  4.3 1   Bo1-2-2-15
        #                             12.9  2.8 5   BR04-C
        #                           11.8    2.6 6   C012-15.85
        #                           11.5    2.6 7   C01gt15.85
        #                 7.6   1.7 14  MT1
        #                 16.2  3.5 15  MT6
        #                           15.6    3.4 8   N04-BR
        # 14.5  3.1 9   N08-BR
        # 19.7  4.1 10  N11-Cgt15.85
        #                           11.5    2.6 11  N14-BR
        # 13.0  2.8 12  N14-C-gt15
        # 40.4  7.2 4   R01c-2-15
        # 57.1  10.3    2   RioG-1-2-15


    MT1_E = [7.6,1.7]
    MT6_E = [16.2,3.5]
    N14br_E = [11.5, 2.6]
    N04br_E = [15.6,3.4]
    C01_E = [11.8,2.6]
    if col_overwrite == 'br_E':
        br_E_dflt_dct = dict_val_overwrite
    else:
        br_E_dflt_dct = {'MT120':MT1_E,  'MT130':MT1_E, 'MT140':MT1_E,  'MT150':MT6_E,'MT160':MT6_E, 'MT170':MT6_E, 'NQT0':N14br_E, 'NQCC2':N14br_E,'NQCV2':N14br_E, 'NQCV4':N04br_E, 'NQPR0':C01_E}

    samp_dflt_dct = {'MT120':'AZ',  'MT130':'AZ', 'MT140':'AZ',  'MT150':'AZ','MT160':'AZ', 'MT170':'AZ', 'NQT0':'SP', 'NQCC2':'SP','NQCV2':'SP', 'NQCV4':'SP', 'NQPR0':'SP'}
    count = 0



    for key in br_E_dflt_dct:
        # make new dict, then append as df to big df
        sampreg = df['sample_region'].loc[df['sample_id'] == key].copy().to_list()[0]
        dct = {'K': K_sed[sampreg], 'D': D_dflt_dct[sampreg],'DF':DF_dflt_dct[sampreg], 'p_re':p_re,  'p_br':p_br, 'coarse_area': coarse_area, 'max_coarse_residence_time':max_coarse_residence_time,'p_crs':p_crs, 'coarse_mass': coarse_mass, 'br_E_rate': br_E_dflt_dct[key][0], 'br_E_rate_unc': br_E_dflt_dct[key][1], 'zcol': 'local_sd_avg' }


        dft = pd.DataFrame(dct, index = [key])
        dft['sample_id']=key
        dft['sample_region'] = samp_dflt_dct[key]
        if count == 0:
            dfdflt = dft.copy()
        else:
            dfdflt = pd.concat([dfdflt, dft])
        count +=1
    return dfdflt


def write_to_excelf(df, saveloc, fp = (5,3), sr = 3):
    with pd.ExcelWriter(saveloc + '_Tables.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        # Add a header format.
        shn = 'all'
        df.to_excel(writer, sheet_name =shn, index = False, startrow = sr, freeze_panes=fp)
        worksheet = writer.sheets[shn]
        worksheet = inner_condfmt(worksheet,workbook, df, sr = sr)
        # Write the column headers with the defined format.
        # for col_num, value in enumerate(df.columns.values):
        #     worksheet.write(sr, col_num, value, header_format)

def get_hexes(numrgb = 10, return_dfx = False,flag_lighter= False, flag_darker = False, flag_rgb = False,low_rthresh = 0, high_rthresh = 255, low_gthresh = 0, high_gthresh = 255, low_bthresh = 0, high_bthresh = 255 ):
    import matplotlib.colors
    # print('beginning')
    xkc = matplotlib.colors.XKCD_COLORS
    xkcd = list(xkc.keys())
    rgba = np.array([matplotlib.colors.to_rgba_array(x)  for x in xkcd])
    rgba = rgba.reshape(rgba.shape[0],4)

    if flag_lighter:
        # print('lighter')
        val = 80/255
        rx = [np.min([x+val, 1]) for x in rgba[:,0]]
        gx =[np.min([x+val, 1]) for x in rgba[:,1]]
        bx = [np.min([x+val, 1]) for x in rgba[:,2]]
        # rx = [np.min([x*1.2, 1]) for x in rgba[:,0]]
        # gx =[np.min([x*1.2, 1]) for x in rgba[:,1]]
        # bx = [np.min([x*1.2, 1]) for x in rgba[:,2]]   # alpha is all ones...
    elif flag_darker:
        # print('darker')
        val = 40/255

        rx =  [np.max([x-val, 0]) for x in rgba[:,0]]
        gx = [np.max([x-val, 0]) for x in rgba[:,0]]
        bx = [np.max([x-val, 0]) for x in rgba[:,0]]
        # rx =  [np.max([x*0.8, 1]) for x in rgba[:,0]]
        # gx = [np.max([x*0.8, 1]) for x in rgba[:,0]]
        # bx = [np.max([x*0.8, 1]) for x in rgba[:,0]]
    else:
        rx = rgba[:,0]
        gx = rgba[:,1]
        bx = rgba[:,2]
    ax = rgba[:,3]

    # want hexx to reflect adjusted Rgb...

    hexx = []
    for i, v in enumerate(rx):
        ctr = (rx[i], gx[i], bx[i])

        if i == 0:
            hexx = [hextriplet(ctr)] # ["#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b)) ]
            # print(rgba[i,:], '\n', ctr, '\n', hexx)
        else:
            hexx.append(hextriplet(ctr))
    # hexx =  ["#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))  for (r,g,b) in (rx,gx,bx)]   # Excel doesn't support hex with alpha collors
    xd = {'xkcd':xkcd, 'r': rx,'g':gx, 'b':bx, 'a':ax, 'hex': hexx}
    dfx = pd.DataFrame(xd)

    dfx = dfx.sort_values(by = ['r', 'g', 'b', 'a']).copy()

    if flag_rgb:
        # print('flag rgb = true')
        dfx = dfx[(dfx['r'] >= low_rthresh/255) & (dfx['r']<=high_rthresh/255)].copy()
        dfx = dfx[(dfx['g'] >= low_gthresh/255) & (dfx['g']<=high_gthresh/255)].copy()
        dfx = dfx[(dfx['b'] >= low_bthresh/255) & (dfx['b']<=high_bthresh/255)].copy()
    hxn = len(dfx['hex'].to_list())
    print('len final filtered df: ', hxn)

    s3 = int(np.max([int((hxn-1)/numrgb), 1]))

    dfx_hxlist = dfx['hex'].to_list()[0::s3]
    print(dfx['xkcd'].to_list()[0::s3])
    print('\tr:', dfx['r'].to_list()[0::s3])
    print('\tr:',dfx['g'].to_list()[0::s3])
    print('\tr:',dfx['b'].to_list()[0::s3])
    # print(' int: ',  s3)

    # print(dfx_hxlist)

    if return_dfx:
        return dfx_hxlist, dfx
    else:
        return dfx_hxlist

def clamp(x):
  return max(0, min(x, 1))

def hextriplet(colortuple):
    return '#' + ''.join(f'{int(i*255):02X}' for i in colortuple)


def inner_condfmt(worksheet,workbook, df, sr = 3):
    LD = range(len(df.columns.to_list()))
    zd = dict(zip(df.columns.to_list(), LD))
    dfo = df.data
    dfl = len(dfo['sample_id'].iloc[:])
    # print(matplotlib.colors.cnames["blue"])
    import matplotlib.colors as mpc
    hx = get_hexes(flag_lighter= True, flag_rgb = True,
    low_rthresh = 100, high_rthresh = 250, low_gthresh = 50, high_gthresh = 80, low_bthresh = 50, high_bthresh = 150 ) #['#04d8b2', '#1e9167', '#3b638c', '#54ac68', '#6832e3', '#7a5901', '#894585', '#985e2b', '#a5fbd5', '#b1d27b', '#c0fb2d', '#cb9d06', '#de9dac', '#fcc006']

    format01 = workbook.add_format({"bg_color": hx[0] })  # light blue, str(mpc.cnames["lightskyblue"])
    format02 = workbook.add_format({"bg_color": hx[1]}) #str(mpc.cnames["steelblue"])})
    format03 = workbook.add_format({"bg_color": hx[2]}) #mpc.cnames["teal"]})
    format04 = workbook.add_format({"bg_color": hx[3]}) #mpc.cnames["forestgreen"]})
    format05 = workbook.add_format({"bg_color": hx[4]}) #mpc.cnames["limegreen"]})
    format06 = workbook.add_format({"bg_color": hx[5]}) #mpc.cnames["yellowgreen"]})
    format07 = workbook.add_format({"bg_color": hx[6]}) #mpc.cnames["yellow"]})
    format08 = workbook.add_format({"bg_color": hx[7]}) #mpc.cnames["gold"]})
    format09 = workbook.add_format({"bg_color": hx[8]}) #mpc.cnames["orange"]})
    format10 = workbook.add_format({"bg_color": hx[9]}) #mpc.cnames["orangered"]})


    format_low= workbook.add_format({"bg_color": mpc.cnames["dimgray"]})

    # format1 = workbook.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})


    list_o_fmt = [format01,format02,format03,format04,format05,format06,format07,format08,format09,format10]
    LB = [0,1,2,3,4,5,6,7,8,9]
    # gradation for normalized fluxes...
    # print(dfl)
    # worksheet.conditional_format(0, 18,dfl+10, 23, {"type": "cell", "criteria": ">=", "value": 1, "format": format1})

    fncols = ['F_coarse_normbr_val', 'F_fines_from_br_normbr_val','F_fines_boxmodel_normbr_val', 'F_dust_normbr_val', 'F_dissolved_normbr_val']
    for i, col in enumerate(fncols):

        for j, c_fm in enumerate(list_o_fmt):
        # j = 0
            worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col],
                {"type": "cell",
                "criteria": "between",
                "minimum": LB[j]/10.,
                "maximum": LB[j]/10.+ 0.099999999999,
                "format": c_fm, },
                )

        worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col], {"type": "cell", "criteria": "<", "value": 0, "format": format_low})



    # Colorise headers -- low rgb vals are darker
    hx = get_hexes( numrgb = 13, flag_lighter= True,flag_rgb = True,low_rthresh = 50, high_rthresh = 100, low_gthresh = 100, high_gthresh = 190, low_bthresh = 90, high_bthresh = 220 ) #['#04d8b2', '#1e9167', '#3b638c', '#54ac68', '#6832e3', '#7a5901', '#894585', '#985e2b', '#a5fbd5', '#b1d27b', '#c0fb2d', '#cb9d06', '#de9dac', '#fcc006']
    format_headergrp00= workbook.add_format({ "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp0= workbook.add_format({"bg_color":hx[0], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp1= workbook.add_format({"bg_color":hx[1], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp2= workbook.add_format({"bg_color":hx[2], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp3= workbook.add_format({"bg_color":hx[3], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp4= workbook.add_format({"bg_color":hx[4], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp5= workbook.add_format({"bg_color":hx[5], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp6= workbook.add_format({"bg_color":hx[6], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp7= workbook.add_format({"bg_color": mpc.cnames["dimgray"], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp8= workbook.add_format({"bg_color":hx[7], "bold": True, "text_wrap": True, "valign": "top", "border": 1})

    hx2 = get_hexes( numrgb = 7, flag_lighter= True,flag_rgb = True,low_rthresh = 80, high_rthresh = 170, low_gthresh = 170, high_gthresh = 240, low_bthresh = 90, high_bthresh = 160 )
    # format_headergrp9= workbook.add_format({"bg_color":hx2[8], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp9= workbook.add_format({"bg_color": mpc.cnames["burlywood"] , "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp10= workbook.add_format({"bg_color":mpc.cnames["wheat"] , "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp11= workbook.add_format({"bg_color":mpc.cnames["lemonchiffon"], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp12= workbook.add_format({"bg_color":mpc.cnames["palegoldenrod"], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    format_headergrp13= workbook.add_format({"bg_color":mpc.cnames["lightgoldenrodyellow"], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    # format_headergrp10= workbook.add_format({"bg_color":hx2[9], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    # format_headergrp11= workbook.add_format({"bg_color":hx2[10], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    # format_headergrp12= workbook.add_format({"bg_color":hx2[11], "bold": True, "text_wrap": True, "valign": "top", "border": 1})
    # format_headergrp13= workbook.add_format({"bg_color":hx2[12], "bold": True, "text_wrap": True, "valign": "top", "border": 1})

    list_o_fmt = [format_headergrp00, format_headergrp0,format_headergrp1,format_headergrp2,format_headergrp3,format_headergrp4, format_headergrp5, format_headergrp6, format_headergrp7,format_headergrp8]

    format_btwn= workbook.add_format({"bg_color":matplotlib.colors.cnames["khaki"]})



    c1 =get_cv_cols_extra1()
    c2 = get_cv_cols_extra2()
    c3 = get_cv_cols_extra3()

    # Color code the headings
    grp0 = ['sample_id'  , 'default_scenario'  ,  'select_col' , 'select_col_val']

    grp1 = ['rt_ky_val','rt_ky_unc']

    grp2 = ['F_fines_boxmodel_g_m2_yr_val','F_coarse_g_m2_yr_val','F_br_g_m2_yr_val', 'F_dust_g_m2_yr_val',   'F_fines_from_br_g_m2_yr_val',  'F_dissolved_g_m2_yr_val', 'F_br_plus_F_dust_g_m2_yr_val',  'F_coarse_plus_F_fines_plus_F_dissolved_g_m2_yr_val', 'F_coarse_plus_F_fines_plus_F_dissolved_g_m2_yr_unc', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr_val']

    grp3 = ['F_coarse_normbr_val', 'F_fines_from_br_normbr_val','F_fines_boxmodel_normbr_val', 'F_dust_normbr_val', 'F_dissolved_normbr_val']

    grp4 = [ 'D_val', 'DF_val', 'p_re_val', 'p_br_val', 'coarse_area_val', 'max_coarse_residence_time_val', 'coarse_mass_val', 'br_E_rate_val','br_E_rate_unc', 'z_val', 'N_val', 'Inv_val', 'Inv_unc' ]

    grp5 =[  'CDF_val', 'CDF_unc', 'K', 'ht_cv_actual_topo','ht_cv_nodust_val', 'ht_cv_BenAsher_val', 'ht_cv_dustnoCDF_val','ht_cv_br_dust_nocdf_val'] + c1

    grp6 =[    'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_max_val', 'dust_10Be_conc_diff_w_gralyAZ_max_val', 'dust_10Be_conc_diff_w_gralySP_unc',  'dust_10Be_conc_diff_w_gralyAZ_unc', 'dust_10Be_conc_diff_w_gralySP_max_unc', 'dust_10Be_conc_diff_w_gralyAZ_max_unc']

    grp7 = [  'Simple_model_apparent_DF_val', 'Simple_model_apparent_DF_unc',
            'DF_if_Fd_is_set_to_1_val', 'DF_if_Fd_is_set_to_1_unc', 'DF_if_Fd_is_set_to_5_val', 'DF_if_Fd_is_set_to_5_unc', 'DF_if_Fd_is_set_to_10_val', 'DF_if_Fd_is_set_to_10_unc', 'DF_if_Fd_is_set_to_15_val', 'DF_if_Fd_is_set_to_15_unc', 'DF_if_Fd_is_set_to_20_val', 'DF_if_Fd_is_set_to_20_unc']

    grp8 = [ 'F_fines_boxmodel_g_m2_yr_unc','F_coarse_g_m2_yr_unc','F_br_g_m2_yr_unc','F_dust_g_m2_yr_unc', 'F_fines_from_br_g_m2_yr_unc','F_dissolved_g_m2_yr_unc', 'F_br_plus_F_dust_g_m2_yr_unc','F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr_unc','F_coarse_normbr_unc', 'F_fines_from_br_normbr_unc','F_fines_boxmodel_normbr_unc', 'F_dust_normbr_unc', 'F_dissolved_normbr_unc', 'sample_region']

    grp9 =  ['ht_cv_actual_minus_dustnoCDFcv_val' ,'ht_cv_actual_minus_noCDFcv_val', 'ht_cv_actual_minus_nodustcv_val','ht_cv_actual_minus_dustcv_val','ht_cv_diff2_nodustcv_dust_cv_val', 'ht_cv_diff2_nodustcv_dustnoCDF_cv_val', 'ht_cv_diff2_dustcv_dustnoCDF_cv_val']
    grplist = [grp0, grp1, grp2, grp3, grp4, grp5, grp6, grp7, grp8,grp9]

    for i, grr in enumerate(grplist):
        for j, col in enumerate(grr):
            # print(sr, zd[col], str(col), list_o_fmt[i])
            if col not in zd.keys():
                print (col)
            else:  # col in zd
                worksheet.write(sr, zd[col], str(col), list_o_fmt[i])
                if i == len(grplist)-1:
                    # apply a format to the ht cv differences. highlight close ones.
                    worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col], {"type": "cell", "criteria": "between", "minimum": -0.003,"maximum": 0.003, "format": format_btwn})

    # Dim out negative valuesin Fluxes
            if i in [2, 3, 6, 7, 8]: # DF if...
                if col in zd.keys():
                    worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col], {"type": "cell", "criteria": "<", "value": 0, "format": format_low})

    list_o_fmt2 = [ format_headergrp9, format_headergrp10, format_headergrp11, format_headergrp12, format_headergrp13]
    # c2 = get_cv_cols_extra2() + get_cv_cols_extra3()
    # for i, nh in enumerate([6,10,18,26,30]):
    #     sk = 'nh'+str(nh)
    #     for j, col in enumerate(c2):
    #         if sk in col:
    #             worksheet.write(sr, zd[col], str(col), list_o_fmt2[i])
    #         if 'unc' not in col:
    #             # apply a format to the ht cv differences. highlight close ones.
    #             worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col], {"type": "cell", "criteria": "between", "minimum": -0.003,"maximum": 0.003, "format": format_btwn})

    # highlight conc although it doesnt' make much sense bc double-counting precip...
    grp6 =[    'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralyAZ_val']
    for j, col in enumerate(grp6):
        worksheet.conditional_format(sr, zd[col],dfl+sr,zd[col], {"type": "cell", "criteria": "between", "minimum": 1.8e7,"maximum": 3.3e8, "format": format_btwn})

    return worksheet
# css_alt_rows = 'background-color: powderblue; color: black;'
# css_indexes = 'background-color: steelblue; color: white;'

# (df.style.apply(lambda col: np.where(col.index % 2, css_alt_rows, None)) # alternating rows
#          .applymap_index(lambda _: css_indexes, axis=0) # row indexes (pandas 1.4.0+)
#          .applymap_index(lambda _: css_indexes, axis=1) # col indexes (pandas 1.4.0+)
# ).to_excel('F:\Ch4_Soils\Tables\column_styled.xlsx', engine='openpyxl')


def get_cv_cols_extra1():
    cols1 = []
    for i,nh in enumerate([6,10,18,26,30]):
        stnh = str(nh)
        c3 = 'ht_cv_actual_topo_nh'+stnh+'m'
        cols1.append(c3)
    return cols1

def get_cv_cols_extra2():
    cols1 = []
    for i,nh in enumerate([6,10,18,26,30]):
        stnh = str(nh)
        c32 = ['ht_cv_actual_minus_nodustcv_nh' + stnh,'ht_cv_actual_minus_dustcv_nh' + stnh,  'ht_cv_actual_minus_dustnoCDFcv_nh' + stnh]
        c4 = [c + '_val' for c in c32]
        [cols1.append(c) for c in c4]
    return cols1

def get_cv_cols_extra3():
    cols1 = []
    for i,nh in enumerate([6,10,18,26,30]):
        stnh = str(nh)
        c32 = ['ht_cv_actual_minus_nodustcv_nh' + stnh,'ht_cv_actual_minus_dustcv_nh' + stnh, 'ht_cv_actual_minus_dustnoCDFcv_nh' + stnh]
        c4u = [c + '_unc' for c in c32]
        [cols1.append(c) for c in c4u]
    return cols1

def format2zero(val):
    return 'number-format: 0'

def format2one(val):
    return 'number-format: 0.0'
def format2two(val):
    return 'number-format: 0.00'
def format2three(val):
    return 'number-format: 0.000'
def format2four(val):
    return 'number-format: 0.0000'


def custom_style_df_allall(df, saveloc, suffixsl = "\\df_allall"):

    c1 =get_cv_cols_extra1()
    c2 = get_cv_cols_extra2()
    c3 = get_cv_cols_extra3()
    colszero = ['K','coarse_area_val', 'max_coarse_residence_time_val', 'coarse_mass_val']
    cols2one = ['z_val', 'br_E_rate_val','br_E_rate_unc','p_re_val', 'p_br_val', 'DF_val','Simple_model_apparent_DF_val', 'Simple_model_apparent_DF_unc', 'DF_if_Fd_is_set_to_1_val', 'DF_if_Fd_is_set_to_1_unc', 'DF_if_Fd_is_set_to_5_val', 'DF_if_Fd_is_set_to_5_unc', 'DF_if_Fd_is_set_to_10_val',  'DF_if_Fd_is_set_to_10_unc', 'DF_if_Fd_is_set_to_15_val', 'DF_if_Fd_is_set_to_15_unc', 'DF_if_Fd_is_set_to_20_val', 'DF_if_Fd_is_set_to_20_unc','rt_ky_val','rt_ky_unc','F_fines_boxmodel_g_m2_yr_val','F_coarse_g_m2_yr_val','F_br_g_m2_yr_val', 'F_dust_g_m2_yr_val',  'F_fines_from_br_g_m2_yr_val',  'F_dissolved_g_m2_yr_val', 'F_br_plus_F_dust_g_m2_yr_val',  'F_coarse_plus_F_fines_plus_F_dissolved_g_m2_yr_val', 'F_coarse_plus_F_fines_plus_F_dissolved_g_m2_yr_unc', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr_val','F_fines_boxmodel_g_m2_yr_unc','F_coarse_g_m2_yr_unc','F_br_g_m2_yr_unc','F_dust_g_m2_yr_unc', 'F_fines_from_br_g_m2_yr_unc','F_dissolved_g_m2_yr_unc', 'F_br_plus_F_dust_g_m2_yr_unc','F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines_g_m2_yr_unc']
    cols2two = ['CDF_val', 'CDF_unc', 'F_coarse_normbr_val', 'F_fines_from_br_normbr_val','F_fines_boxmodel_normbr_val', 'F_dust_normbr_val', 'F_dissolved_normbr_val','F_coarse_normbr_unc', 'F_fines_from_br_normbr_unc','F_fines_boxmodel_normbr_unc', 'F_dust_normbr_unc', 'F_dissolved_normbr_unc']
    cols2three = ['ht_cv_actual_topo',
    'ht_cv_BenAsher_val', 'ht_cv_br_dust_nocdf_val',
    'ht_cv_actual_minus_dustnoCDFcv_val',
    'ht_cv_actual_minus_noCDFcv_val',
    'ht_cv_BenAsher_unc', 'ht_cv_nodust_val', 'ht_cv_nodust_unc','ht_cv_actual_minus_nodustcv_val','ht_cv_actual_minus_dustcv_val', 'ht_cv_dustnoCDF_val', 'ht_cv_diff2_nodustcv_dust_cv_val', 'ht_cv_diff2_nodustcv_dustnoCDF_cv_val', 'ht_cv_diff2_dustcv_dustnoCDF_cv_val'] + c1
    list_o_lists = [colszero, cols2one, cols2two,cols2three]
    list_o_funcs = [format2zero, format2one, format2two, format2three]


    styler = df.style
    for i, fm in enumerate(list_o_funcs):
        # print(list_o_lists[i])
        styler.applymap(fm, subset = list_o_lists[i])

    write_to_excelf(styler, saveloc + suffixsl,fp = [4,4], sr = 3)

