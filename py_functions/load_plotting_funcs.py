
def set_up_df_for_flux_results2(df,dff, Ncol = 'calc-10Be_at_g', N_unc_col = 'calc-uncert_10Be_at-g', I_desc = 'site'):
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

    dff.drop(['sample_region', 'sample_id'], axis = 1, inplace = True)
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

def get_new_df_results_w_unc(df, col_overwrite = False, val_overwrite = 0, flag_br_equiv = False, flag_coarse_subsurface = 10, flag_pde = False):
    # col_overwrite = ['coarse_mass'], val_overwrite= 2000,

    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)

    dff = write_defaults_to_df(df)
    #     print('Default df cols: ', dff.columns.to_list())
    #     print('L4: dff[\'coarse_mass\'].iloc[0]', dff['coarse_mass'].iloc[0])
    #     print('L4: dff[\'coarse_area\'].iloc[0]', dff['coarse_area'].iloc[0])
    if isinstance(col_overwrite, list):  # Then we wanna calculate non-default vals
        for i, ovrc in enumerate(col_overwrite):
            ovrval = val_overwrite[i] # Will write uniform value for all samples
            if ovrc not in dff.columns.to_list():
                print('whoops column to overwrite isnt in the default df, adding col anyways: ', ovrc )
            dff[ovrc] = ovrval

    dft, SD, N = set_up_df_for_flux_results2(df,dff)  # does not output ufloats yet
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
            print(flag_coarse_subsurface, 'flag coarse subsurface is a float or int')
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

    dft['dust_10Be_conc_diff_w_gralySP'] =(dft['D'] - SP_D_graly)/dft['F_dust']
    dft['dust_10Be_conc_diff_w_gralyAZ'] =(dft['D'] - AZ_D_graly)/dft['F_dust']

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

    dft['F_br_solids_after_chem_wx'] = (F_br- F_coarse)/dft['DF']

    dft['F_fines_from_br'] = dft['F_fines_boxmodel'] - dft['F_dust']
    dft['F_dissolved'] = (dft['F_fines_boxmodel'] - dft['F_dust']) * dft['DF']

    # These should be equivalent: LHS = RHS of mass balance
    dft['F_br_plus_F_dust'] = dft['F_br'] + dft['F_dust']
    dft['F_coarse_plus_F_fines_plus_F_dissolved']= dft['F_coarse'] + dft['F_fines_boxmodel'] + dft['F_dissolved']

    DF = dft['DF']
    p_re = dft['p_re']
    dft['K'] = dft.sample_region.map({'AZ':2.6, 'SP':14.6})

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

    dft['CDF'] = (1/DF * (F_br - dft['F_coarse']) + 2.3/2.6*dft['F_coarse'])/F_br  ##  ## Scaling by density ratios,  needs to incorporate coarse sediment as well

    # K is in cm2/yr so it's OK that Fluxes are in g/cm2/yr as well // actually need x100
    # g/cm2/yr / (g/cm3) / (cm2/yr) = cm/yr / (cm2/yr) = 1/cm
    # if ht cv should be in m, then 1/cm * 100 cm/1m = 100 / m
    # p_re should incorporate the coarse sediments as well....
    ht_cv = -(dft['F_br'] + dft['F_dust'])*(1-dft['CDF'] )/p_re / dft['K']

    dft['ht_cv_BenAsher'] = ht_cv*100

    dft['ht_cv_nodust'] = -(dft['F_br'] )*(1-dft['CDF'] )/p_re / dft['K'] * 100

    # Simple mass balance model:
    # Fbr = Fcoarse + Ffines + Fdissolved, ---> Fbr and Fcoarse and Ffines are the same as in more complexs model, but LHS is smaller so RHS also has to be smaller...
    # RHS simple mass balance model:
    dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']  =dft['F_br']  - dft['F_coarse'] - dft['F_fines_boxmodel']

    dft['Simple_model_apparent_DF'] = dft['F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']/ dft['F_fines_boxmodel']

    # Can solve for DF if we assume Fd:
        #
    dft['DF_if_Fd_is_set_to_1'] = (F_coarse - F_br)/(1/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_5'] = (F_coarse - F_br)/(5/10000 - F_fines) -1     # g/m2/yr
    dft['DF_if_Fd_is_set_to_10'] = (F_coarse - F_br)/(10/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_15'] = (F_coarse - F_br)/(15/10000 - F_fines) -1
    dft['DF_if_Fd_is_set_to_20'] = (F_coarse - F_br)/(20/10000 - F_fines) -1

    if flag_pde:
        dft = f_Inv_unc_pde(dft, N,p_re,SD, N_unc, p_re_unc, z_unc)
        dft = f_rest_unc_pde(dft, Inv, D, ltl, Inv_unc, D_unc)
        dft = f_erate_unc_pde(dft, SD, D, N, ltl, p_re, z_unc, D_unc, N_unc, p_re_unc)

    to_m2_cols = ['F_fines_boxmodel', 'F_coarse', 'F_br', 'F_dust', 'F_br_solids_after_chem_wx','F_fines_from_br', 'F_dissolved', 'F_br_plus_F_dust','F_coarse_plus_F_fines_plus_F_dissolved', 'F_dissolved_simple_nodust_F_br_minus_F_coarse_minus_F_fines']

    # Change fluxes to m2
    for c,cc in enumerate(to_m2_cols):
        dft[cc + '_g_m2_yr'] = dft[cc].apply(lambda x: x*10000)

    dft['rt_ky'] = dft['rt'] /1000 # ky

    # At very end, run through all columns and make sure none are Affine Scalar Function
    for c, col in enumerate(dft.columns.to_list()):
        fv = dft[col].iloc[0]
    # Actually I think remaking all ufloats will improve those horribly long decimals        if isinstance(fv, uncertainties.core.Variable):
    #             dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
        if isinstance(fv, uncertainties.core.AffineScalarFunc) | isinstance(fv, uncertainties.core.Variable):
    #             print('Line 96:' , col, dft[col].iloc[0],'\n', dft[col])
            dft = dft.copy()
            dft[col] = redef_uf(dft[col])

            dft[col + '_val'], dft[col + '_unc'] = get_vals_uf(dft[col])
    #             print(col, dft[col].iloc[1])
        else: pass

    return dft

def ufloat_precise(uf, pr = 3):
    # this actually seems to work!
    # v hard to set format for these ufloats. So I'll truncate them to pr precision (sig figs...)
    uf_orig = uf
    print('original uf: ', uf)
    uf_nv = uf.nominal_value
    uf_std = uf.std_dev
    log_val = int(np.floor(np.log10(uf_nv)))
    # print(log_val)
    # print(1-log_val)
    uf_dec = np.round(uf_nv*10**(1-log_val), pr-1)
    uf_std_dec = np.round(uf_std *10**(1-log_val), pr-1)
    # print(uf_dec, 10**log_val)
    # print(uf_std)

    uf_new = ufloat(uf_dec*10**(log_val-1), uf_std_dec*10**(log_val-1))
    print(uf_new)
    return uf_new

def get_default_vars():
    #  p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, zcol, max_coarse_residence_time, br_E_rate_unc, Ncol, N_unc_col, ltl, atom10Bemass, p_br =get_default_vars()
    p_re = 1.4
    DF = 20   # [10,20,50] # Makes v little diff 75 vs 100
    # Vars to actually explore
    br_E_rate = 11.5    # [7.6, 16.2, 11.5, 15.6, 19.7] # mm/kyr
    D_Desc = 'Graly_SP' # ['Graly_SP', 'Graly_AZ','Graly_SP_x4']
    D_sp = 'SP'  # ['SP', 'AZ','SPx4']
    D = SP_D_graly # [SP_D_graly, AZ_D_graly,SP_D_graly*4]
    # default is [0]
    coarse_mass  = 1133 # [1000, 1133,1500]  # cm
    # coarse_mass  = [1133]  # cm
    coarse_area = 300# [300, 500]
    zcol = 'local_sd_avg' # ['local_sd_avg', 15] ## will change
    max_coarse_residence_time = 11*1000  # [8*1000, 11*1000 ] # yr

    br_E_rate_unc = 2.6   # [1.7, 3.5,2.6, 3.4, 4.1]
    Ncol = 'calc-10Be_at_g'
    N_unc_col = 'calc-uncert_10Be_at-g'
    ltl = 5.1e-7 # 1/yr
    atom10Bemass = 1.6634e-23 #g/at
    p_br = 2.6
    Idesc = 'site'
    #     set_all_unc_pct()
    return p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, zcol, max_coarse_residence_time, br_E_rate_unc, Ncol, N_unc_col, ltl, atom10Bemass, p_br, Idesc

def set_all_unc_pct(pct_unc, p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, z, max_coarse_residence_time):
    p_re_unc = p_re *pct_unc/100
    DF_unc = DF *pct_unc/100
    D_unc= D*pct_unc/100
    coarse_mass_unc = coarse_mass*pct_unc/100
    coarse_area_unc= coarse_area *pct_unc/100
    z_unc=z *pct_unc/100
    max_coarse_residence_time_unc = max_coarse_residence_time*pct_unc/100
    return p_re_unc,z_unc,D_unc,DF_unc,coarse_mass_unc,coarse_area_unc,max_coarse_residence_time_unc

def set_all_unc_zero():
    p_re_unc =0
    z_unc =0
    D_unc =0
    DF_unc= 0
    coarse_mass_unc =0
    coarse_area_unc= 0
    max_coarse_residence_time_unc =0
    return p_re_unc,z_unc,D_unc,DF_unc,coarse_mass_unc,coarse_area_unc,max_coarse_residence_time_unc

def modify_start_subsoil_coarse_seds(dft, percent_subsurface_by_vol_is_coarse):
    coarse_mass = dft['coarse_mass'].iloc[0]
    coarse_area= dft['coarse_area'].iloc[0]
    p_re = dft['p_re'].iloc[0]
    SD = dft['z'].copy()
    p_br = dft['p_br'].iloc[0]
    #     print('In modify_start_subsoil_coarse_seds: \ncoarse mass: ', coarse_mass,'\ncoarse area' ,coarse_area, '\np_re: ' , p_re, '\nSD: ', SD, '\np_br', p_br)
    SDnew = SD-SD*percent_subsurface_by_vol_is_coarse/100   # h_fines
    coarse_massnew = coarse_mass+SD*percent_subsurface_by_vol_is_coarse/100*p_br*coarse_area   # equivalent mass of subsurface clasts
    #p_renew = some kind of table Poesen
    return SDnew, coarse_massnew
def D_graly(P,L, flag_P_is_mm_per_yr = True):
    # Graly 2010a
    # latitude = L
    # P = precip in cm/yr
    # flux in 10^4 at/cm3  but elsewhere in the paper it's written at/cm2/yr....
    if flag_P_is_mm_per_yr:
        P = P/10
    flux = P*(1.44/(1+np.exp((30.7-L)/4.36))+0.63)*10**4
    return flux

def write_defaults_to_df(df):
    # Rules:
    # D --> if MT, then Graly AZ, else Graly SP
    AZ_D_graly = D_graly(400, 31.2)
    SP_D_graly = D_graly(510, 39.1)
    # AZ_D_graly = D_graly(450, 31.2)
    # SP_D_graly = D_graly(450, 39.1)

    D_dflt_dct = {'AZ': AZ_D_graly, 'SP': SP_D_graly}
    # DF --> based on cao soil/cao br; SP: ~4% cao in soil so 25 DF, MT: ~ 6% cao in soil so ~ 17 DF. BUT some soils in both samples have ~ 20% cao AND if include MgO then MT 9, 30%, and SP is 7,8, 26%. But I think these are probably sm rock fragments elevating these vals.... so use lowest
    DF_dflt_dct = {'AZ': 17, 'SP': 25}
    # p_re --> 1.4
    # p_re, p_br, coarse_area, max_coarse_residence_time, p_crs, coarse_mass
    p_re = 1.4
    # p_br --> 2.6
    p_br = 2.6
    # coarse area
    coarse_area = 300
    # max coarse residence time.
    max_coarse_residence_time = 15500 # 11000
    # density of coarse sediment modify? but if anything I want to overestimate the coarse sedments SHOULD be 2.2 or so (Poesen Laveee 1994)
    p_crs = 2.2
    # coarse mass
    coarse_mass = 1133
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
    br_E_dflt_dct = {'MT120':MT1_E,  'MT130':MT1_E, 'MT140':MT1_E,  'MT150':MT6_E,'MT160':MT6_E, 'MT170':MT6_E, 'NQT0':N14br_E, 'NQCC2':N14br_E,'NQCV2':N14br_E, 'NQCV4':N04br_E, 'NQPR0':C01_E}
    samp_dflt_dct = {'MT120':'AZ',  'MT130':'AZ', 'MT140':'AZ',  'MT150':'AZ','MT160':'AZ', 'MT170':'AZ', 'NQT0':'SP', 'NQCC2':'SP','NQCV2':'SP', 'NQCV4':'SP', 'NQPR0':'SP'}
    count = 0
    for key in br_E_dflt_dct:
        # make new dict, then append as df to big df
        sampreg = df['sample_region'].loc[df['sample_id'] == key].copy().to_list()[0]
        dct = {'D': D_dflt_dct[sampreg],'DF':DF_dflt_dct[sampreg], 'p_re':p_re,  'p_br':p_br, 'coarse_area': coarse_area, 'max_coarse_residence_time':max_coarse_residence_time,'p_crs':p_crs, 'coarse_mass': coarse_mass, 'br_E_rate': br_E_dflt_dct[key][0], 'br_E_rate_unc': br_E_dflt_dct[key][1], 'zcol': 'local_sd_avg' }
        dft = pd.DataFrame(dct, index = [key])
        dft['sample_id']=key
        dft['sample_region'] = samp_dflt_dct[key]
        if count == 0:
            dfdflt = dft.copy()
        else:
            dfdflt = pd.concat([dfdflt, dft])
        count +=1
    return dfdflt

def add_literature_dust_conc(ax, xl= 'None'):
    plt.sca(ax)
    if isinstance(xl, str):
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xl[0], xl[1]
    ax.add_patch(mpatches.Rectangle( (xmin, 1.8e7),xmax-xmin, 1.07e8-1.8e7, facecolor="grey",alpha = 0.6, zorder = 1, label = 'Graly 2010a Dust Concentration (at/g), New Zealand'))
    plt.axhline(3.3e8,c='dimgray', zorder = 1, label = 'Mendez-Garc{\'}ia 2023 Dust Concentration (at/g), Altzomoni MX')

def wrap_default_vars():
    # p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, zcol, max_coarse_residence_time,br_E_rate_unc,  Ncol, N_unc_col, ltl, atom10Bemass, p_br, Idesc, p_re_unc, z_unc, D_unc, DF_unc, coarse_mass_unc, coarse_area_unc, max_coarse_residence_time_unc, br_E_rate_i, max_coarse_residence_time_i, coarse_area_i, coarse_mass_i, DF_i, D_i, p_re_i = wrap_default_vars()
    p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, zcol, max_coarse_residence_time,br_E_rate_unc,  Ncol, N_unc_col, ltl, atom10Bemass, p_br, Idesc =get_default_vars()
    p_re_unc,z_unc,D_unc,DF_unc,coarse_mass_unc,coarse_area_unc,max_coarse_residence_time_unc = set_all_unc_zero()
    # need to make ufloats: br_E_rate_i, max_coarse_residence_time_unc_i, coarse_area_i, coarse_mass_i, DF_i, D_i, p_re_i
    br_E_rate_i = ufloat(br_E_rate, br_E_rate_unc,"br E")
    max_coarse_residence_time_i = ufloat(max_coarse_residence_time, max_coarse_residence_time_unc, "max coarse rt")
    coarse_area_i = ufloat(coarse_area, coarse_area_unc,"coarse area")
    coarse_mass_i = ufloat(coarse_mass, coarse_mass_unc,"coarse mass")
    DF_i = ufloat(DF, DF_unc, "DF")
    D_i = ufloat(D, D_unc, "D")
    p_re_i = ufloat(p_re, p_re_unc, "p_re")
    return p_re,DF, br_E_rate,D_Desc,D_sp, D, coarse_mass, coarse_area, zcol, max_coarse_residence_time,br_E_rate_unc,  Ncol, N_unc_col, ltl, atom10Bemass, p_br, Idesc, p_re_unc, z_unc, D_unc, DF_unc, coarse_mass_unc, coarse_area_unc, max_coarse_residence_time_unc, br_E_rate_i, max_coarse_residence_time_i, coarse_area_i, coarse_mass_i, DF_i, D_i, p_re_i

def plot_default_vals(ax,dfo, xcol, row_selector_col = 'sample_region', row_selector_val = 'AZ'):
    dff = write_defaults_to_df(dfo)
    dft = dff[dff[row_selector_col] == row_selector_val].copy()
    plt.sca(ax)
    plt.axvline(dft[xcol].iloc[0], c = 'lightgray', lw = 6, label = 'default '+row_selector_col + ' value: ' + str(row_selector_val))
    return ax

def savefig(filenametag, saveloc, h_all, l_all, figsize,w_legend = True, prefixtag = 'flux_plots_', leg_cols = 1):
    fig = plt.gcf()
    sn = '\\' +prefixtag+ filenametag + '_wo_legend'
    savep = saveloc + sn + '.png'
    fig.savefig(savep,dpi=300, bbox_inches='tight')
    savep = saveloc + sn + '.svg'
    fig.savefig(savep,dpi=300, bbox_inches='tight')
    if w_legend:
        fig.set_size_inches([figsize[0]+3, figsize[1]])
        fig.legend(handles = h_all, labels = l_all, loc = 2, bbox_to_anchor = (1.05,.99), ncols = leg_cols)

        sn = '\\' +prefixtag+ filenametag + '_w_legend'
        savep = saveloc + sn + '.png'
        fig.savefig(savep,dpi=300, bbox_inches='tight')
        savep = saveloc + sn + '.svg'
        fig.savefig(savep,dpi=300, bbox_inches='tight')
    print(savep)

def plot_fluxes_original(df1,dfo, filenametag,xc = 's_dx (cm)', xaxis = 'Sample distance from ridgetop (m)' ,xcf= .01,unitmod = 10000,  Fyaxis_desc = ['Fine fraction', 'Dust', 'Coarse fraction', 'Bedrock', 'Bedrock contribution to \nsoil after chemical wx'], ycols = ['F_fines_boxmodel_g_m2_yr_val', 'F_dust_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_br_g_m2_yr_val', 'F_br_solids_after_chem_wx_g_m2_yr_val'],  unccols = ['F_fines_boxmodel_g_m2_yr_unc','F_dust_g_m2_yr_unc', 'F_coarse_g_m2_yr_unc', 'F_br_g_m2_yr_unc', 'F_br_solids_after_chem_wx_g_m2_yr_unc'],  yll = [[0,30], [0,30], [0,30],[0,40]], figsize = [10,8],xll = 'auto',  colorcol = 'sample_id',  plot_column_col = 'sample_region', br_Ecol= 'br_E_rate_val', sz = 30,mm = ['o', 's', 'd', '^', 'p', '>', '<'],units = 'g/m$^2$/yr', saveloc = 'F:\Ch4_Soils\Figures_draftdata\\', xlog = True, ylog = False):

    # need flag for separate plots by sample_region or sample_id  F_br_solids_after_chem_wx
    nrows = 5
    ncols = len(df1[plot_column_col].unique())
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex=True)
  # 'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralySP_unc', 'dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralyAZ_unc'
    h_all = []
    l_all = []
    for sr, sampr in enumerate(df1[plot_column_col].unique()):
        dft11 = df1[df1[plot_column_col] == sampr].copy()

        for i, sa in enumerate(dft11[colorcol].unique()):  # loop through each sample name, each is a color from dict
            dftemp = dft11[dft11[colorcol] == sa].copy()

            cc = dftemp['cdict'].iloc[0]
            for j, fx in enumerate(ycols):  # loop through each row
                plt.sca(ax[j, sr])
                if xll != 'auto':
                    plt.xlim(xll)
                if j == len(ycols)-1:
                    if i == 0:
                        if xc != 'select_col_val':
                            plot_default_vals(ax[j, sr],dfo, xc[:-4], row_selector_col = plot_column_col, row_selector_val = sampr)
                        else:
                            pass #                             plot_default_vals(ax[j, sr],dfo, xc[:-4], row_selector_col = plot_column_col, row_selector_val = sampr)

                    if sampr == 'AZ':
                        fx =  'dust_10Be_conc_diff_w_gralyAZ_val'
                    else:
                        fx = 'dust_10Be_conc_diff_w_gralySP_val'

                if yll != 'auto':
                    plt.ylim(yll[j])

                if xlog: plt.xscale('log')
                if ylog: plt.yscale('log')

                if i == 0:
                    L1 = Fyaxis_desc[j] + ', ' + sa
                else: L1= ''

                if sr == 0:
                    if j< len(ycols)-1:plt.ylabel('{:s} \nSediment Flux\n ({:s})'.format(Fyaxis_desc[j],units))
                    else: plt.ylabel('[$^{10}$Be] of dust (at/g)')
                else:
                    if yll != 'auto':
                        ax[j, sr].axes.yaxis.set_ticklabels([])


                plt.sca(ax[j, sr])
    #                 print(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], dftemp[unccols[j]]*unitmod[j])
                plt.errorbar(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], yerr = dftemp[unccols[j]]*unitmod[j], c = 'gray', zorder = 1,ls='none')
                h1 = plt.scatter(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], sz,marker = mm[j], c =cc, label =L1, zorder = 10)
                h_all.append(h1)


                if j == len(ycols)-1:
                    plt.xlabel(xaxis)
                    if yll == 'auto':
                        yl1 = ax[j, sr].get_ylim()
                        ax[j, sr].set_ylim([0, yl1[1]])

                if j == len(ycols)-2: #br contribution to fines
                    if i == 0: L1 = '(Bedrock Flux - Coarse Fraction)/ DF, E = {:.1f} mm/ky'.format(dftemp[br_Ecol].iloc[0])
                    else: L1 = ''
    #                     for jj in [2,3]:
                    plt.sca(ax[j, sr])
                    fx = ycols[j+1] #
                    plt.errorbar(dftemp[xc]*xcf,dftemp[fx]*unitmod[j] , yerr = dftemp[unccols[j]]*unitmod[j],xerr = 0, c = 'gray', zorder = 1,ls='none')
                    h2 = plt.scatter(dftemp[xc]*xcf,dftemp[fx]*unitmod[j] , sz,marker = mm[j+1], c =cc, label =L1, zorder = 10)
                    h_all.append(h2)

    #     yl1 = ax[len(ycols)-1, 0].get_ylim()
    #     yl2 = ax[len(ycols)-1, 1].get_ylim()
    #     for aa in [0,1]:
    #         ax[len(ycols)-1, aa].set_ylim([min(yl1[0], yl2[0]), max(yl1[1], yl2[1])])
    #     print(xc, dftemp[xc], dftemp[xc]*xcf)
    fig.set_size_inches(figsize)
    plt.tight_layout()
    savefig(filenametag, saveloc, h_all, l_all, figsize)
    print('Uncertainties represent only the cosmo chemistry carried forward ')

def run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = False, ylog = False, flag_only_regional_plots = True):
    figsize = [8,6]
    if xll == 'auto':
        xll = [0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]
    # dfDF['select_col'].iloc[0] + '_val'
    plot_fluxes(dfDF,df, filenametag =filenametag, xc = xc,plot_column_col = 'sample_region', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize, xlog = xlog, ylog = ylog, saveloc = saveloc2)
    #
    if not flag_only_regional_plots:
        filenametag = filenametag + '_samples_sep'
        saveloc2 = saveloc + dirn
        figsize = [20,8]
        plot_fluxes(dfDF,df, filenametag =filenametag, xc = xc,plot_column_col = 'sample_id', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize,  xlog = xlog, ylog = ylog, saveloc = saveloc2)


        filenametag = filenametag + '_samples_sep_yaxis_indiv'
        saveloc2 = saveloc + dirn
        yllu = 'auto'
        plot_fluxes(dfDF,df, filenametag =filenametag, xc = xc,plot_column_col = 'sample_id', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize, xlog = xlog, ylog = ylog, saveloc = saveloc2)

def plot_fluxes(df1,dfo, filenametag,xc = 's_dx (cm)', xaxis = 'Sample distance from ridgetop (m)' ,xcf= .01,unitmod = 10000,
    Fyaxis_desc = ['Bedrock','Coarse fraction','Dust',  'Bedrock contribution to \nsoil after chemical wx', 'Dissolved'],
    ycols = ['F_br_g_m2_yr','F_coarse_g_m2_yr','F_dust_g_m2_yr', 'F_fines_from_br_g_m2_yr','F_dissolved_g_m2_yr'],
     yll = [[0,30], [0,30], [0,30],[0,40]], figsize = [10,8],xll = 'auto',  colorcol = 'sample_id',
     plot_column_col = 'sample_region', br_Ecol= 'br_E_rate_val', sz = 30,mm = ['o', 's', 'd', '^', 'p', '>', '<'],
     units = 'g/m$^2$/yr', saveloc = 'F:\Ch4_Soils\Figures_draftdata\\', xlog = True, ylog = False):
    # plot fluxes only
    # need flag for separate plots by sample_region or sample_id
    unccols = [v+'_unc' for v in ycols]
    ycols = [v + '_val' for v in ycols]
    nrows = len(ycols)
    ncols = len(df1[plot_column_col].unique())
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex=True)
  # 'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralySP_unc', 'dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralyAZ_unc'
    h_all = []
    l_all = []
    for sr, sampr in enumerate(df1[plot_column_col].unique()):
        dft11 = df1[df1[plot_column_col] == sampr].copy()

        for i, sa in enumerate(dft11[colorcol].unique()):  # loop through each sample name, each is a color from dict
            dftemp = dft11[dft11[colorcol] == sa].copy()

            cc = dftemp['cdict'].iloc[0]
            for j, fx in enumerate(ycols[0:-1]):  # loop through each row
                plt.sca(ax[j, sr])
                if xll != 'auto':
                    plt.xlim(xll)
                if j == len(ycols)-1:
                    if i == 0:
                        if xc != 'select_col_val':
                            plot_default_vals(ax[j, sr],dfo, xc[:-4], row_selector_col = plot_column_col, row_selector_val = sampr)
                            print('what')



                if xlog: plt.xscale('log')
                if ylog: plt.yscale('log')

                if j == 0:
                    L1 = Fyaxis_desc[j] + '          ' + sa
                elif i == 0 and sr == 1:
                    L1 = Fyaxis_desc[j]
                else:
                    L1 = ''

                if sr == 0:
                    plt.ylabel('{:s} \nSediment Flux\n ({:s})'.format(Fyaxis_desc[j],units))
                # else:
                    # if yll != 'auto':
                    #     ax[j, sr].axes.yaxis.set_ticklabels([])
                if yll != 'auto':
                    plt.sca(ax[j, sr])
                    plt.ylim(yll[j])

                plt.sca(ax[j, sr])
    #                 print(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], dftemp[unccols[j]]*unitmod[j])
                xxx, x_std = get_vals_uf(dftemp[xc])
                yyy, y_std = get_vals_uf(dftemp[fx])
                xux, xus = get_vals_uf(dftemp[unccols[j]])
                plt.errorbar(xxx*xcf, yyy*unitmod[j], yerr =xux*unitmod[j], c = 'gray', zorder = 1,ls='none')
                h1 = plt.scatter(xxx*xcf, yyy*unitmod[j], sz,marker = mm[j], c =cc, label =L1, zorder = 10)
                h_all.append(h1)
                # print('xc in plot_fluxes: ', xc,'\t xcf: ', xcf, '\t fx', fx, '\t unitmod', unitmod[j])


    #             if j == nrows-1:
    #                 plt.xlabel(xaxis)
    #                 # br contribution to fines
    #                 if j == 0 and i == 0 and sr == 1: L1 = '(Bedrock Flux - Coarse Fraction)/ DF, E = {:.1f} mm/ky'.format(dftemp[br_Ecol].iloc[0])
    #                 else: L1 = ''
    # #                     for jj in [2,3]:
    #                 plt.sca(ax[j, sr])
    #                 fx = ycols[j+1] #


    #                 xxx, x_std = get_vals_uf(dftemp[xc])
    #                 yyy, y_std = get_vals_uf(dftemp[fx])
    #                 xux, xus = get_vals_uf(dftemp[unccols[j]])

    #                 plt.errorbar(xxx*xcf,yyy*unitmod[j] , yerr = xux*unitmod[j],xerr = 0, c = 'gray', zorder = 1,ls='none')
    #                 h2 = plt.scatter(xxx*xcf,yyy*unitmod[j], sz,marker = mm[j+1], c =cc, label =L1, zorder = 10)
    #                 h_all.append(h2)
    for j, fx in enumerate(ycols[0:-1]):
            ylt = ax[j, 0].get_ylim()
            ylt2 = ax[j, 1].get_ylim()
            for sr in [0,1]:
                ax[j, sr].set_ylim([0,np.max([ylt2[1], ylt[1]])*1.1])

    fig.set_size_inches(figsize)
    plt.tight_layout()
    savefig(filenametag, saveloc, h_all, l_all, figsize)
    # print('Uncertainties represent only the cosmo chemistry carried forward ')
def run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xll = 'auto',  yllu = [[1e3,1e11], [1, 1e5], [0.0001, .1], [0.0001, .1]], xlog = False, ylog = False, xc = 'auto', flag_only_regional_plots = True):
    figsize = [8,6]
    if xll == 'auto':
        xll = [0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]
    if xc == 'auto':
        xc =dfDF['select_col'].iloc[0] + '_val'
    # dfDF['select_col'].iloc[0] + '_val'

    filenametag = filenametag + '_rt'

    dfDF['ht_cv_actual_topo_abs'] = np.abs(dfDF['ht_cv_actual_topo_nh18m'] )
    dfDF['ht_cv_BenAsher_abs'] = np.abs(dfDF['ht_cv_BenAsher'] )
    plot_other_vals(dfDF,df, Fyaxis_desc = ['$^{10}$Be Conc. \nin dust (at/g)', 'Residence time \n(fine fraction) \n(ky)', 'Hilltop Cv - Measured', 'Hilltop Cv- Ben Asher (2019) prediction'],
                    ycols = ['rt_ky_val','ht_cv_actual_topo_abs', 'ht_cv_BenAsher_abs'],
                    unccols = [ 'rt_ky_unc'],
                    filenametag =filenametag, xc = xc,plot_column_col = 'sample_region', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize, xlog = xlog, ylog = ylog, saveloc = saveloc2, nrows = 3)
    if not flag_only_regional_plots:
        filenametag = filenametag + '_samples_sep'
        saveloc2 = saveloc + dirn
        figsize = [20,8]
        plot_other_vals(dfDF,df, Fyaxis_desc = ['$^{10}$Be Conc. \nin dust (at/g)', 'Residence time \n(fine fraction) \n(ky)', 'Hilltop Cv - Measured', 'Hilltop Cv- Ben Asher (2019) prediction'],
                        ycols = ['rt_ky_val','ht_cv_actual_topo_abs', 'ht_cv_BenAsher_abs'],
                        unccols = [ 'rt_ky_unc'],
                        filenametag =filenametag, xc = xc,plot_column_col = 'sample_id', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize, xlog = xlog, ylog = ylog, saveloc = saveloc2, nrows = 3)

        filenametag = filenametag + '_yaxis_indiv'
        saveloc2 = saveloc + dirn
        yllu = 'auto'
        plot_other_vals(dfDF,df, Fyaxis_desc = ['$^{10}$Be Conc. \nin dust (at/g)', 'Residence time \n(fine fraction) \n(ky)', 'Hilltop Cv - Measured', 'Hilltop Cv- Ben Asher (2019) prediction'],
                        ycols = ['rt_ky_val','ht_cv_actual_topo_abs', 'ht_cv_BenAsher_abs'],
                        unccols = [ 'rt_ky_unc'],
                        filenametag =filenametag, xc = xc,plot_column_col = 'sample_id', xaxis = xlabel ,xcf= 1, xll = xll, unitmod = [1,1,1,1,1], yll = yllu, figsize = figsize, xlog = xlog, ylog = ylog, saveloc = saveloc2, nrows = 3)

def plot_other_vals(df1,dfo, filenametag,xc = 's_dx (cm)', xaxis = 'Sample distance from ridgetop (m)' ,xcf= .01,unitmod = 10000,  Fyaxis_desc = ['Fine fraction', 'Dust', 'Coarse fraction', 'Bedrock', 'Bedrock contribution to \nsoil after chemical wx'], ycols = ['F_fines_boxmodel_g_m2_yr_val', 'F_dust_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_br_g_m2_yr_val', 'F_br_after_chem_wx_g_m2_yr_val'],  unccols = ['F_fines_boxmodel_g_m2_yr_unc','F_dust_g_m2_yr_unc', 'F_coarse_g_m2_yr_unc', 'F_br_g_m2_yr_unc', 'F_br_after_chem_wx_g_m2_yr_unc'],  yll = [[0,30], [0,30], [0,30],[0,40]], figsize = [10,8],xll = 'auto',  colorcol = 'sample_id',  plot_column_col = 'sample_region', br_Ecol= 'br_E_rate_val', sz = 30,mm = ['o', 's', 'd', '^', 'p', '>', '<'],units = 'g/m$^2$/yr', saveloc = 'F:\Ch4_Soils\Figures_draftdata\\', xlog = True, ylog = False, nrows = 3):
    # plot fluxes only
    # need flag for separate plots by sample_region or sample_id

    ncols = len(df1[plot_column_col].unique())
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex=True)
  # 'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralySP_unc', 'dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralyAZ_unc'
    h_all = []
    l_all = []
    axhcount = 0
    for sr, sampr in enumerate(df1[plot_column_col].unique()):
        dft11 = df1[df1[plot_column_col] == sampr].copy()

        for i, sa in enumerate(dft11[colorcol].unique()):  # loop through each sample name, each is a color from dict
            dftemp = dft11[dft11[colorcol] == sa].copy()
            cc = dftemp['cdict'].iloc[0]
            sample_reg = dftemp['sample_region'].iloc[0]

            for j in range(len(ycols)+1):  # loop through each data to plot; 2 ht cv ones
                # unc flag :
                if j in [0, 1]:
                    flag_unc = True
                    axj = j
                else:
                    flag_unc = False
                    axj = 2
                # if j in [0, 1]: ylog = True
                # else: ylog = False
                ylog = True

                plt.sca(ax[axj, sr])

                if xll != 'auto':
                    plt.xlim(xll)
                if j == len(ycols)-1:
                    if i == 0:
                        if (xc != 'select_col_val') and (xc != 'zcol'):
                            plot_default_vals(ax[axj, sr],dfo, xc[:-4], row_selector_col = plot_column_col, row_selector_val = sampr)
                            # print('what')

                if xlog: plt.xscale('log')
                if ylog: plt.yscale('log')

                if j == 0:
                    L1 = Fyaxis_desc[j] + '          ' + sa
                elif i == 0 and sr == 1:
                    L1 = Fyaxis_desc[j]
                else:
                    L1 = ''

                if sr == 0:
                    plt.ylabel('{:s}'.format(Fyaxis_desc[axj]))
                # else:
                    # if yll != 'auto':
                    #     ax[j, sr].axes.yaxis.set_ticklabels([])

                plt.sca(ax[axj, sr])
                if sample_reg == 'AZ' and j == 0: # AZ
                    fx ='dust_10Be_conc_diff_w_gralyAZ_val'
                    fxu = 'dust_10Be_conc_diff_w_gralyAZ_unc'
                    if axhcount == 0:
                        plt.axhline(1.8e7, c = 'green', lw = 6, label = 'Graly 2010a, NZ dust \n[$^{10}$Be]')
                        plt.axhline(1.07e8, c = 'green', lw = 6, label = 'Graly 2010a, NZ dust \n[$^{10}$Be], upper bound')
                        plt.axhline(3.3e8, c = 'yellowgreen', lw = 6, label = 'Mendez-García, Altzomoni in MX \n[$^{10}$Be]')

                elif sample_reg == 'SP' and j == 0:
                    fx = 'dust_10Be_conc_diff_w_gralySP_val'
                    fxu = 'dust_10Be_conc_diff_w_gralySP_unc'
                    if axhcount == 0:
                        plt.axhline(1.8e7, c = 'green', lw = 6, label = 'Graly 2010a, NZ dust \n[$^{10}$Be]')
                        plt.axhline(1.07e8, c = 'green', lw = 6, label = 'Graly 2010a, NZ dust \n[$^{10}$Be], upper bound')
                        plt.axhline(3.3e8, c = 'yellowgreen', lw = 6, label = 'Mendez-García, Altzomoni in MX \n[$^{10}$Be]')

                elif j == 1:
                    fx = ycols[j-1]
                    fxu = unccols[j-1]
                    plt.axhline(11, c = 'lightgray', lw = 6, label = 'Coarse Sediments Maximum \nResidence Time (ky)')

                elif j in [2,3]:
                    fx = ycols[j-1]
                    flag_unc = False

                else:
                    print('j, sr: ', j, sr, 'not sure what to do about fx')



                axhcount += 1
                if yll != 'auto':
                    # print(yll, j)
                    # print('yll not auto: ', yll[j] , ' for fx: ', fx)
                    plt.ylim(yll[j])


                xxx, x_std = get_vals_uf(dftemp[xc])
                yyy, y_std = get_vals_uf(dftemp[fx])
                xus, xu_std = get_vals_uf(dftemp[fxu])

                                # print(xc)
                if flag_unc:
                    plt.errorbar(xxx*xcf, yyy*unitmod[j], yerr = xus*unitmod[j], c = 'gray', zorder = 1,ls='none')

                # xxx = redef_uf(dftemp[xc])
                # yyy = redef_uf(dftemp[fx])
                # print(xxx, yyy, xcf, unitmod[j], sz, mm[j], cc, L1)
                # print(type(xxx), type(yyy), type(xcf), type(unitmod[j]), type(sz), mm[j], cc, L1)
                if j == 2 and i == 0:  # ht cv measured as horiz line
                    h1 = plt.axhline(yyy[0]*unitmod[j],color = 'k', label =L1)
                else:
                    h1 = plt.scatter(xxx*xcf, yyy*unitmod[j], sz,marker = mm[j], c =cc, label =L1, zorder = 10)
                h_all.append(h1)

                if j == nrows-1:
                    plt.xlabel(xaxis)

    # for j, fx in enumerate(ycols[0:-1]):
    #         ylt = ax[axj, 0].get_ylim()
    #         ylt2 = ax[axj, 1].get_ylim()
    #         for sr in [0,1]:
    #             ax[axj, sr].set_ylim([0,np.max([ylt2[1], ylt[1]])*1.1])

    fig.set_size_inches(figsize)
    plt.tight_layout()
    savefig(filenametag, saveloc, h_all, l_all, figsize)
    # print('Uncertainties represent only the cosmo chemistry carried forward ')



def plot_ht_cv_predicted(df1,dfo, filenametag,xc = 's_dx (cm)', xaxis = 'Sample distance from ridgetop (m)' ,xcf= .01,unitmod = 10000,  Fyaxis_desc = ['Actual hilltop curvature', 'Predicted hilltop curvature with est. dust flux (Ben-Asher 2019)'], ycols = ['ht_cv_actual_topo', 'ht_cv_BenAsher' ], figsize = [6,6],xll = [0, 20],  colorcol = 'sample_id',  br_Ecol= 'br_E_rate_val', sz = 30,mm = ['o', 's', 'd', '^', 'p', '>', '<'],units = 'g/m$^2$/yr', saveloc = 'F:\Ch4_Soils\Figures_draftdata\\', xlog = True, ylog = False):

    # need flag for separate plots by sample_region or sample_id
    nrows = 1
    ncols = 1
    fig, ax = plt.subplots(ncols = ncols, nrows = nrows, sharex=True)
  # 'dust_10Be_conc_diff_w_gralySP_val', 'dust_10Be_conc_diff_w_gralySP_unc', 'dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralyAZ_unc'
    h_all = []
    l_all = []

    dft = df1.dropna(subset = ycols[0])
    for i, sa in enumerate(df1[colorcol].unique()):  # loop through each sample name, each is a color from dict
        dftemp = df1[df1[colorcol] == sa].copy()

        cc = dftemp['cdict'].iloc[0]
        for j, fx in enumerate(ycols):  # loop through each row

            plt.sca(ax[j, sr])
            if i == 0:
                plt.title(sa)
            plt.xlim(xll)
            if j == len(ycols)-1:
                plt.yscale('log')
                if i == 0:
                    plot_default_vals(ax[j, sr],dfo, xc[:-4], row_selector_col = plot_column_col, row_selector_val = sampr)

                if sampr == 'AZ':
                    fx =  'dust_10Be_conc_diff_w_gralyAZ_val'
                else:
                    fx = 'dust_10Be_conc_diff_w_gralySP_val'

            plt.ylim(yll[j])

            if xlog: plt.xscale('log')
            if ylog: plt.yscale('log')

            if i == 0:
                L1 = Fyaxis_desc[j] + ', ' + sa
            else: L1= ''

            if sr == 0:
                if j< len(ycols)-1:plt.ylabel('{:s} \nSediment Flux\n ({:s})'.format(Fyaxis_desc[j],units))
                else: plt.ylabel('[$^{10}$Be] of dust (at/g)')
            else:
                ax[j, sr].axes.yaxis.set_ticklabels([])

            if j == len(ycols)-1:
                plt.xlabel(xaxis)

            plt.sca(ax[j, sr])
            #                 print(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], dftemp[unccols[j]]*unitmod[j])
            plt.errorbar(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], yerr = dftemp[unccols[j]]*unitmod[j], c = 'gray', zorder = 1,ls='none')
            h1 = plt.scatter(dftemp[xc]*xcf, dftemp[fx]*unitmod[j], sz,marker = mm[j], c =cc, label =L1, zorder = 10)
            h_all.append(h1)
            if j == len(ycols)-2: #br contribution to fines
                if i == 0: L1 = '(Bedrock Flux - Coarse Fraction)/ DF, E = {:.1f} mm/ky'.format(dftemp[br_Ecol].iloc[0])
                else: L1 = ''
                #                     for jj in [2,3]:
                plt.sca(ax[j, sr])
                fx = ycols[j+1] #
                plt.errorbar(dftemp[xc]*xcf,dftemp[fx]*unitmod[j] , yerr = dftemp[unccols[j]]*unitmod[j],xerr = 0, c = 'gray', zorder = 1,ls='none')
                h2 = plt.scatter(dftemp[xc]*xcf,dftemp[fx]*unitmod[j] , sz,marker = mm[j+1], c =cc, label =L1, zorder = 10)
                h_all.append(h2)

    fig.set_size_inches(figsize)
    plt.tight_layout()
    savefig(filenametag, saveloc, h_all, l_all)
    print('Uncertainties represent only the cosmo chemistry carried forward')

def write_just_latex(df,  sheetname, saveloc,col_order,  dict_of_fmt = None, cwdth = 50):
    sn = saveloc +'\\' + sheetname + '.txt'
    cf = "p{{{:.0f}pt}}".format(cwdth)*len(df.columns.to_list())
    df = df[col_order].copy()
    df.style.format(dict_of_fmt).format_index( axis=1,escape="latex" ).hide(axis=0).to_latex(sn, column_format=cf, position="h!", position_float="centering", hrules=True, label="tab:out", caption='Caption')

def edit_latex_tbl(ffs):
    # print(ffs)
    with open(ffs) as f:
        ftxt = f.read()
        ftxt = ftxt.replace(r'+/-0)', r' ').replace(r'+/-0 &', r' &').replace(r'+/-0 \\', r' \\').replace(r'+/-0', r'$\pm$').replace(r'+/-', r'$\pm$').replace(r'$\pm$.', r'$\pm$0.').replace(r'e-0', r'e-').replace(r'e+0', 'e').replace(r'e+', 'e').replace(r'(', r' ').replace(r')', r' ').replace('_dust', '$_{dust}$')
        ftxt = ftxt.replace(r'\textbackslash Be\{\{\}\}', r'\Be{}').replace(r'\$\_\{', r'$_{').replace(r'\}\$', r'}$').replace(r'\$\textasciicircum 2\$', r'$^2$').replace(r'\$', r'$').replace(r'\_', r'_')
        ftxt = ftxt.replace(r' cm ', r'(cm)').replace(r' yr ', r'(yr)').replace(r' ky ', r'(ky)').replace(r' mm/kyr', r'(mm/ky)').replace(r' g/m$^2$/yr ', '(g/m$^2$/yr)')
        # strip latex table stuff above and below the data
        ftxt = ftxt.replace(r'\begin{table}[h!]', '').replace(r'\centering', '').replace(r'\caption{Caption}', '').replace(r'\label{tab:out}', '').replace(r'\end{table}', '')
    print(ftxt,  file=open(ffs[0:-4]+ '.txt', 'w+'))

def write_to_excel(df, saveloc):
    with pd.ExcelWriter(saveloc + '_Tables.xlsx', engine='xlsxwriter') as writer:
        workbook = writer.book
        # Add a header format.
        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "border": 1,
            }
        )
        shn = 'all'
        df.to_excel(writer, sheet_name =shn, index = False)
        worksheet = writer.sheets[shn]
        worksheet.freeze_panes(1, 1)

        # Write the column headers with the defined format.
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)

def select_df_rows(df1):
    #     df1, cdict =select_df_rows(df1)
    df1 = df1[df1.sample_id.isin(['MT120','MT130','MT140', 'MT150','MT160', 'MT170', 'NQT0', 'NQCV2', 'NQCV4','NQPR0', 'NQCC2'])].copy()
    return df1
def map_cdict(df):
    cdict = {'MT120': 'peru',
     'MT130': 'indianred',
     'MT140': 'coral',
     'MT150': 'orange',
     'MT160': 'gold',
     'MT170': 'yellowgreen',
     'NQT0': 'black',
     'NQCC2': 'darkturquoise',
     'NQCV2': 'mediumaquamarine',
     'NQCV4': 'teal',
     'NQPR0': 'mediumblue'}
    df['cdict'] = df.sample_id.map(cdict)
    return df

def wrap_newdf(dfto,K = 14.6, print_df_short = False, summary_val_name = 'none',  col_overwrite = 'none', vals_arr = 0, flag_br_equiv = False, flag_coarse_subsurface = False, flag_pde = False, tn = 'TableName',saveloc = 'F:\Ch4_Soils\Tables', dirn = '\\24_26_01', dict_of_fmt = None):
    if summary_val_name != 'none':
        col = summary_val_name
        print('col is summary_val_name')
    else:
        col = col_overwrite[0]

    for i, val in enumerate(vals_arr):
        if flag_coarse_subsurface:
            dft = get_new_df_results_w_unc(dfto, col_overwrite = col_overwrite, val_overwrite = [val], flag_br_equiv = flag_br_equiv, flag_coarse_subsurface = val, flag_pde = flag_pde)
        else:
            dft = get_new_df_results_w_unc(dfto, col_overwrite = col_overwrite, val_overwrite = [val], flag_br_equiv = flag_br_equiv, flag_coarse_subsurface = flag_coarse_subsurface, flag_pde = flag_pde)
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
    if not os.path.exists(saveloc +  dirn):
        os.mkdir(saveloc+ dirn)
    # also need to write xlsx
    sn =  saveloc+dirn +'\\' + tn
    write_to_excel(dfn, sn)

    # Also need script to write Latex txt
    latex_col_order = pc
    # reduce rows for Latex and plots -- only samples we care about.
    dfn =select_df_rows(dfn)

    write_just_latex(dfn, tn, saveloc+dirn,latex_col_order , dict_of_fmt = dict_of_fmt)
    sn = saveloc+dirn +'\\' + tn  + '.txt'
    edit_latex_tbl(sn)

    # also need to make + save plots.
    dfn = map_cdict(dfn)
    #     print(dfn.columns.to_list())

    # colors are now columns
    return dfn
