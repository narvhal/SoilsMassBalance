

def run_all_vars(df, saveloc,dff_overwrite, dff_val, Dcol_overwrite, dcol_dict_val, dirn, saveloc2, plotstuff = False):

    # Coarse seds

    vals_arr =  [0,25,50,75,99]
    col_overwrite = 'D'   # not doing anything, as long as it's not isinstance, list, I think.

    table_name = 'CoarseSeds_subsurface_pct'
    summary_val_name = 'Coarse_seds_subsurface'
    printdfshort = False
    if not os.path.exists(saveloc):
        os.mkdir(saveloc)
    flag_coarse_subsurface = True

    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)

    df_all = dfDF.copy()

    # yllu is unique to each run
    yllu = [[0,25],[0,25],[0,25],[0,55]]
    filenametag = 'CoarseSeds_subsurface_pct_varies'
    xlabel = 'Percent subsurface \n>2mm diam (%)'
    xc = 'select_col_val'
    xll = [0.8*dfDF['select_col_val'].min(), 1.2*dfDF['select_col_val'].max()]
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = False, ylog = False)

    xc = 'select_col_val'

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xlog = False, xc = xc)

    # D variations
    vals_arr = [ AZ_D_graly*.1, AZ_D_graly*.5, AZ_D_graly,AZ_D_graly*5, AZ_D_graly*10, AZ_D_graly*50 ]
    col_overwrite = ['D']
    table_name = 'D_variations'
    summary_val_name = 'none'
    printdfshort = False
    # ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    # ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    # yllu = [[0,ym],[0,ym],[0,ym],[0,ym2]]
    figsize = [10,10]
    filenametag = 'D_varies'

    # yllu_rt = [[1e3,1e11], [.1, 1e5], [0.00001, 0.1], [0.000001, 0.1]]

    # yllu is unique to each run
    yllu = [[0,25],[0,25],[0,25],[0,55]]
    saveloc2 = saveloc + dirn
    xlabel = 'Delivery rate (at/cm$^2$/yr)'
    flag_coarse_subsurface = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)

    df_all = pd.concat([df_all, dfDF])

    xc = dfDF['select_col'].iloc[0] + '_val'
    xll = [0.8*dfDF['select_col_val'].min(), 1.2*dfDF['select_col_val'].max()]

    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll,yllu,  saveloc2, xlog = True, ylog = False)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xlog = True)


    # vals_arr = [ AZ_D_graly*.05, AZ_D_graly*.1, AZ_D_graly*.5, AZ_D_graly,AZ_D_graly*5 ]
    # col_overwrite = ['D']
    # table_name = 'D_variations_zoom'
    # summary_val_name = 'none'
    # printdfshort = False


    # dfDF =  wrap_newdf2(df, tn = table_name,
    #                     dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
    #                     print_df_short = printdfshort, summary_val_name = summary_val_name,
    #                     col_overwrite =col_overwrite, vals_arr = vals_arr,
    #                     flag_coarse_subsurface = flag_coarse_subsurface,
    #                     saveloc = saveloc,dirn = dirn)
    # df_all = pd.concat([df_all, dfDF])

    # ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    # ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    # ymc = np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.6, 2)

    # yllu = [[0,ym2],[0,ym2],[0,ym2],[0,ym2], [0,1e10]]
    # figsize = [10,10]
    # filenametag = 'D_varies_zoom'
    # saveloc2 = saveloc + dirn

    # if plotstuff:
    #     run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = True, ylog = True)

    # # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # # for second row, rt, use ycols[0], and unccols[0]
    # # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    # if plotstuff:
    #     run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xlog = True)


    # DF var

    vals_arr =  [1,3,5, 10, 15, 20, 35]
    col_overwrite = ['DF']
    table_name = 'DF_variations'
    summary_val_name = 'none'
    printdfshort = False

    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    ymc = np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.6, 2)

    yllu = 'auto' # [[0,ym],[0,ym],[0,ym],[0,ym2], [0,ymc]]
    figsize = [10,10]
    filenametag = 'DF_varies'
    saveloc2 = saveloc + dirn
    xlabel = 'Dissolution Factor'

    xlog = False
    xll = [min(vals_arr)*0.9, max(vals_arr)*1.1]
    print(xll)
    xc = 'select_col_val'
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = False)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xlog = xlog)


    # p_re var
    vals_arr =  [0.5,1, 1.4, 1.8]
    col_overwrite = ['p_re']
    table_name = 'p_re_variations'
    summary_val_name = 'none'
    printdfshort = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    ymc = np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.1, 2)
    # print(ymc)
    yllu = 'auto'   # [[0,ym],[0,ym],[0,ym],[0,ym2], [0,ymc]]
    figsize = [10,10]
    filenametag = 'p_re_varies'
    saveloc2 = saveloc + dirn
    xlabel = '(Fine) Soil Density'

    xlog = False
    xll = [min(vals_arr)*0.9, max(vals_arr)*1.1]

    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = True)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xll, xlog = xlog)

    # Bedrock erosion rate variations
    vals_arr =  [5, 10, 20, 50, 70]
    col_overwrite = ['br_E_rate']
    table_name = 'Br_E_rates_var'
    summary_val_name = 'none'
    printdfshort = False
    flag_coarse_subsurface = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    ymc = [np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].min().min() *0.8, 0),
    np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.6, 0)]

    yllu ='auto' #  [[0,ym],[0,ym],[0,ym],[0,ym2], ymc]
    figsize = [10,10]
    filenametag = table_name
    saveloc2 = saveloc + dirn
    xlabel = 'Bedrock Erosion rate (mm/ky)'
    xc = 'select_col_val'
    xll = 'auto'#[0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]


    xlog = False
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = True)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2,xlog = xlog)

    # soil depth


    vals_arr =  [5, 10, 20, 50, 70]
    col_overwrite = ['zcol']
    table_name = 'zcol_var'
    summary_val_name = 'none'
    printdfshort = False
    flag_coarse_subsurface = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    ymc = [np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].min().min() *0.8, 0),
    np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.6, 0)]

    yllu ='auto' #  [[0,ym],[0,ym],[0,ym],[0,ym2], ymc]
    figsize = [10,10]
    filenametag = table_name
    saveloc2 = saveloc + dirn
    xlabel = 'Soil Depth (cm)'
    xc = 'zcol'
    xll = 'auto'#[0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]
    xlog = False
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = True)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2, xc= xc,xlog = xlog)



# Coarse mass


    vals_arr =  [1000, 1500, 2000]
    col_overwrite = ['coarse_mass']
    table_name = 'coarse_mass_var'
    summary_val_name = 'none'
    printdfshort = False
    flag_coarse_subsurface = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    ym = np.round(dfDF[['F_fines_boxmodel_g_m2_yr_val', 'F_coarse_g_m2_yr_val', 'F_dust_g_m2_yr_val']].max().max() *1.1, 1)
    ym2 = np.round(dfDF['F_br_g_m2_yr_val'].max().max() *1.1, 1)
    ymc = [np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].min().min() *0.8, 0),
    np.round(dfDF[['dust_10Be_conc_diff_w_gralyAZ_val', 'dust_10Be_conc_diff_w_gralySP_val']].max().max() *1.6, 0)]

    yllu ='auto' #  [[0,ym],[0,ym],[0,ym],[0,ym2], ymc]
    figsize = [10,10]
    filenametag = table_name
    saveloc2 = saveloc + dirn
    xlabel = 'Coarse sediment mass (g)'
    xc = 'select_col_val'
    xll = 'auto'#[0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]
    xlog = False
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = True)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2, xc= xc,xlog = xlog)


# Coarse mass RT


    vals_arr =  [7000, 11000, 15500, 20000]
    col_overwrite = ['max_coarse_residence_time']
    table_name = 'max_coarse_residence_time_var'
    summary_val_name = 'none'
    printdfshort = False
    flag_coarse_subsurface = False


    dfDF =  wrap_newdf2(df, tn = table_name,
                        dff_overwrite = dff_overwrite, dff_val = dff_val, Dcol_overwrite = Dcol_overwrite, dcol_dict_val = dcol_dict_val,
                        print_df_short = printdfshort, summary_val_name = summary_val_name,
                        col_overwrite =col_overwrite, vals_arr = vals_arr,
                        flag_coarse_subsurface = flag_coarse_subsurface,
                        saveloc = saveloc,dirn = dirn)
    df_all = pd.concat([df_all, dfDF])

    yllu ='auto' #  [[0,ym],[0,ym],[0,ym],[0,ym2], ymc]
    figsize = [10,10]
    filenametag = table_name
    saveloc2 = saveloc + dirn
    xlabel = 'Coarse Sediment Max Residence Time'
    xc = 'select_col_val'
    xll = 'auto'#[0.9*dfDF['select_col_val'].min(), 1.1*dfDF['select_col_val'].max()]
    xlog = False
    if plotstuff:
        run_plots_fluxes(dfDF, df, filenametag, xc, xlabel, xll, yllu, saveloc2,xlog = xlog, ylog = True)

    # for first row, conc in dust, need to use diff column name for each plot column (az, SP), and assoc. uncert
    # for second row, rt, use ycols[0], and unccols[0]
    # for third row, ht cv, use ycols[1:], no unc, but label with K, E-rate, and fraction of dust or dust flux.
    if plotstuff:
        run_plots_other_vals(dfDF, df, filenametag, xlabel, saveloc2, xc= xc,xlog = xlog)






    write_to_excel(df_all,saveloc + dirn + '\df_all')


    dfdflt = write_defaults_to_df2(df)

    write_to_excel(dfdflt, saveloc2 + r'\defaults')

    latex_col_order = dfdflt.columns.to_list()[-2:] + dfdflt.columns.to_list()[0:-3]
    tn = r'\defaults'
    write_just_latex(dfdflt, r'defaults', saveloc,latex_col_order)
    sn = saveloc+'\\' + tn  + '.txt'
    edit_latex_tbl(sn)


    plt.close('all')
    return saveloc + dirn + '\df_all'





