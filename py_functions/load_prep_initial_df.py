#load_prep_initial_df

def prep_initial_df(df):
    cols = ['calc-uncert_10Be_at-g', 'calc-10Be_at_g', 'inv_sum_flag', 'sample_region', 'sample_location', 'sample_position', 'sample_top', 'sample_bot', 'sample_vol_cm', 'sample_type', 'sample_id', 'sample_is_transect', 'sample_is_profile', 'local_soildepth1','local_soildepth2', 'local_soildepth3', 'local_soildepth4','local_soildepth_avg', 's_lat', 's_lon', 's_elev', 's_elev_src', 'slpcombined', 's_cv', 's_dx', 's_dx (cm)', 'Inventory (at/cm2) each site', 'cv_prof_sm8m', 'cv_prof_sm30m','cv_tot_sm30m', 'gradient_sm8_in_degrees', 's_dx (cm) upslope', 'Inventory each site_localdepth', 'depth_avg_conc_each_site', 'Inventory[at/cm2]']
    dft = df[cols].copy()
    # local soildepth min/max
    loc_sd = ['local_soildepth1','local_soildepth2', 'local_soildepth3', 'local_soildepth4']
    loc_sd_agg  = ['local_sd_avg', 'local_sd_min', 'local_sd_max']

    df['local_sd_avg'] = df[loc_sd].mean(axis=1)
    df['local_sd_min'] = df[loc_sd].min(axis=1)
    df['local_sd_max'] = df[loc_sd].max(axis=1)
    # df[['local_sd_avg','local_sd_min', 'local_sd_max']]
    df['local_sd_avg'] = df['local_sd_avg'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    df['local_sd_min'] = df['local_sd_min'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    df['local_sd_max'] = df['local_sd_max'].fillna(df['local_soildepth_avg']).fillna(df['sample_vol_cm'])
    # df[['local_sd_avg','local_sd_min', 'local_sd_max']]
    return df
