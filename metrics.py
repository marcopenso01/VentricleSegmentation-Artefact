import logging
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import binary_metric as bm
import configuration as config
import scipy.stats as stats
import utils

logging.basicConfig(
    level=logging.INFO  # allow DEBUG level messages to pass through the logger
)


def conv_int(i):
    return int(i) if i.isdigit() else i


def natural_order(sord):
    """
    Sort a (list,tuple) of strings into natural order.
    Ex:
    ['1','10','2'] -> ['1','2','10']
    ['abc1def','ab10d','b2c','ab1d'] -> ['ab1d','ab10d', 'abc1def', 'b2c']
    """
    if isinstance(sord, tuple):
        sord = sord[0]
    return [conv_int(c) for c in re.split(r'(\d+)', sord)]


def print_latex_tables(df, eval_dir, circle=False):
    """
    Report geometric measures in latex tables to be used in the ACDC challenge paper.
    Prints mean (+- std) values for Dice for all structures.
    :param df:
    :param eval_dir:
    :return:
    """
    if not circle:
        out_file = os.path.join(eval_dir, 'latex_tables.txt')
    else:
        out_file = os.path.join(eval_dir, 'circle_latex_tables.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 1\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # prints mean (+- std) values for Dice, all structures, averaged over both phases.

        header_string = ' & '
        line_string = 'METHOD '

        for s_idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            for measure in ['dice']:

                header_string += ' & {} ({}) '.format(measure, struc_name)

                dat = df.loc[df['struc'] == struc_name]

                if measure == 'dice':
                    line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                else:
                    line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

            if s_idx < 2:
                header_string += ' & '
                line_string += ' & '

        header_string += ' \\\\ \n'
        line_string += ' \\\\ \n'

        text_file.write(header_string)
        text_file.write(line_string)

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 2\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # table 2: mean (+- std) values for Dice and HD, all structures, both phases separately

        for idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['dice', 'hd']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['struc'] == struc_name)]

                    if measure == 'dice':
                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.2f}\,({:.2f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('table 3\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')
        # table 3: mean (+- std) values for Recall and Precision, all structures, both phases separately

        for idx, struc_name in enumerate(['LV', 'RV', 'Myo']):
            # new line
            header_string = ' & '
            line_string = '({}) '.format(struc_name)

            for p_idx, phase in enumerate(['ED', 'ES']):
                for measure in ['recall', 'prec']:

                    header_string += ' & {} ({}) '.format(phase, measure)

                    dat = df.loc[(df['phase'] == phase) & (df['struc'] == struc_name)]

                    if measure == 'recall':
                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))
                    else:
                        line_string += ' & {:.3f}\,({:.3f}) '.format(np.mean(dat[measure]), np.std(dat[measure]))

                if p_idx == 0:
                    header_string += ' & '
                    line_string += ' & '

            header_string += ' \\\\ \n'
            line_string += ' \\\\ \n'

            if idx == 0:
                text_file.write(header_string)

            text_file.write(line_string)

    return 0


def compute_metrics_on_directories_raw(input_fold, output_fold, dice=True):
    '''
    - Dice
    - Hausdorff distance
    - Sens, Spec, F1
    - Predicted volume
    - Volume error w.r.t. ground truth
    :return: Pandas dataframe with all measures in a row for each prediction and each structure
    '''
    if dice:
        data = h5py.File(os.path.join(input_fold, 'pred_on_dice.hdf5'), 'r')
    else:
        data = h5py.File(os.path.join(input_fold, 'pred_on_loss.hdf5'), 'r')

    cardiac_phase = []
    file_names = []
    structure_names = []

    # measures per structure:
    dices_list = []
    hausdorff_list = []
    prec_list = []
    sens_list = []
    vol_list = []
    vol_err_list = []
    vol_gt_list = []

    dices_cir_list = []
    hausdorff_cir_list = []
    prec_cir_list = []
    sens_cir_list = []
    vol_cir_list = []
    vol_err_cir_list = []

    structures_dict = {1: 'RV', 2: 'Myo', 3: 'LV'}

    for paz in np.unique(data['paz'][:]):
        ind = np.where(data['paz'][:] == paz)

        for ph in np.unique(data['phase'][:]):

            pred_arr = []  # predizione del modello
            mask_arr = []  # ground truth
            cir_arr = []  # predizione circle

            for i in range(len(ind[0])):
                if data['phase'][ind[0][i]] == ph:
                    pred_arr.append(data['pred'][ind[0][i]])
                    mask_arr.append(data['mask'][ind[0][i]])
                    cir_arr.append(data['mask_cir'][ind[0][i]])

            pred_arr = np.transpose(np.asarray(pred_arr, dtype=np.uint8), (1, 2, 0))
            mask_arr = np.transpose(np.asarray(mask_arr, dtype=np.uint8), (1, 2, 0))
            cir_arr = np.transpose(np.asarray(cir_arr, dtype=np.uint8), (1, 2, 0))

            for struc in [3, 1, 2]:
                gt_binary = (mask_arr == struc) * 1
                pred_binary = (pred_arr == struc) * 1
                cir_binary = (cir_arr == struc) * 1

                # vol[ml] = n_pixel * (x_dim*y_dim) * z_dim / 1000
                # 1 mm^3 = 0.001 ml
                volpred = pred_binary.sum() * (0.6467 * 0.6467) * config.z_dim / 1000.
                volgt = gt_binary.sum() * (0.6467 * 0.6467) * config.z_dim / 1000.
                volcir = cir_binary.sum() * (0.6467 * 0.6467) * config.z_dim / 1000

                vol_list.append(volpred)  # volume predetto CNN
                vol_cir_list.append(volcir)  # volume predetto circle
                vol_err_list.append(volpred - volgt)
                vol_err_cir_list.append(volcir - volgt)
                vol_gt_list.append(volgt)  # volume reale

                # Dice
                temp_dice = 0
                temp_cir_dice = 0
                count = 0
                for zz in range(gt_binary.shape[2]):
                    slice_pred = np.squeeze(pred_binary[:, :, zz])
                    slice_gt = np.squeeze(gt_binary[:, :, zz])
                    slice_cir = np.squeeze(cir_binary[:, :, zz])

                    if slice_gt.sum() == 0 and slice_pred.sum() == 0:
                        temp_dice += 1
                    elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
                        temp_dice += 0
                    elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
                        temp_dice += bm.dc(pred_binary, gt_binary)

                    if slice_gt.sum() == 0 and slice_cir.sum() == 0:
                        temp_cir_dice += 1
                    elif slice_cir.sum() == 0 and slice_gt.sum() > 0:
                        temp_cir_dice += 0
                    elif slice_cir.sum() != 0 and slice_gt.sum() != 0:
                        temp_cir_dice += bm.dc(cir_binary, gt_binary)
                    count += 1
                dices_list.append(temp_dice/count)
                dices_cir_list.append(temp_cir_dice/count)

                # Hausdorff distance
                hd_max = 0
                hd_cir_max = 0
                for zz in range(gt_binary.shape[2]):
                    slice_pred = np.squeeze(pred_binary[:, :, zz])
                    slice_gt = np.squeeze(gt_binary[:, :, zz])
                    slice_cir = np.squeeze(cir_binary[:, :, zz])

                    if slice_gt.sum() == 0 and slice_pred.sum() == 0:
                        hd_value = 0
                    elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
                        hd_value = 1
                    elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
                        hd_value = bm.hd(slice_gt, slice_pred, (0.6467, 0.6467), connectivity=2)

                    if slice_gt.sum() == 0 and slice_cir.sum() == 0:
                        hd_cir_value = 0
                    elif slice_cir.sum() == 0 and slice_gt.sum() > 0:
                        hd_cir_value = 1
                    elif slice_cir.sum() != 0 and slice_gt.sum() != 0:
                        hd_cir_value = bm.hd(slice_gt, slice_cir, (0.6467, 0.6467), connectivity=2)

                    if hd_max < hd_value and hd_value < 20:
                        hd_max = hd_value
                    if hd_cir_max < hd_cir_value:
                        hd_cir_max = hd_cir_value

                #recall
                temp_rec = 0
                temp_cir_rec = 0
                count = 0
                for zz in range(gt_binary.shape[2]):
                    slice_pred = np.squeeze(pred_binary[:, :, zz])
                    slice_gt = np.squeeze(gt_binary[:, :, zz])
                    slice_cir = np.squeeze(cir_binary[:, :, zz])

                    if slice_gt.sum() == 0 and slice_pred.sum() == 0:
                        temp_rec += 1
                    elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
                        temp_rec += 0
                    elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
                        temp_rec += bm.recall(pred_binary, gt_binary)

                    if slice_gt.sum() == 0 and slice_cir.sum() == 0:
                        temp_cir_rec += 1
                    elif slice_cir.sum() == 0 and slice_gt.sum() > 0:
                        temp_cir_rec += 0
                    elif slice_cir.sum() != 0 and slice_gt.sum() != 0:
                        temp_cir_rec += bm.recall(cir_binary, gt_binary)
                    count += 1
                sens_list.append(temp_rec / count)
                sens_cir_list.append(temp_cir_rec / count)

                #precision
                temp_prec = 0
                temp_cir_prec = 0
                count = 0
                for zz in range(gt_binary.shape[2]):
                    slice_pred = np.squeeze(pred_binary[:, :, zz])
                    slice_gt = np.squeeze(gt_binary[:, :, zz])
                    slice_cir = np.squeeze(cir_binary[:, :, zz])

                    if slice_gt.sum() == 0 and slice_pred.sum() == 0:
                        temp_prec += 1
                    elif slice_pred.sum() == 0 and slice_gt.sum() > 0:
                        temp_prec += 0
                    elif slice_pred.sum() != 0 and slice_gt.sum() != 0:
                        temp_prec += bm.precision(pred_binary, gt_binary)

                    if slice_gt.sum() == 0 and slice_cir.sum() == 0:
                        temp_cir_prec += 1
                    elif slice_cir.sum() == 0 and slice_gt.sum() > 0:
                        temp_cir_prec += 0
                    elif slice_cir.sum() != 0 and slice_gt.sum() != 0:
                        temp_cir_prec += bm.precision(cir_binary, gt_binary)
                    count += 1
                prec_list.append(temp_prec / count)
                prec_cir_list.append(temp_cir_prec / count)

                hausdorff_list.append(hd_max)
                hausdorff_cir_list.append(hd_cir_max)
                cardiac_phase.append(ph)
                file_names.append(paz)
                structure_names.append(structures_dict[struc])

    # CNN
    df1 = pd.DataFrame({'dice': dices_list, 'hd': hausdorff_list,
                        'vol': vol_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_list,
                        'phase': cardiac_phase, 'struc': structure_names, 'filename': file_names,
                        'recall': sens_list, 'prec': prec_list})

    # Circle
    df2 = pd.DataFrame({'dice': dices_cir_list, 'hd': hausdorff_cir_list,
                        'vol': vol_cir_list, 'vol_gt': vol_gt_list, 'vol_err': vol_err_cir_list,
                        'phase': cardiac_phase, 'struc': structure_names, 'filename': file_names,
                        'recall': sens_cir_list, 'prec': prec_cir_list})

    data.close()
    return df1, df2


def print_stats(df, eval_dir, circle=False):
    if not circle:
        out_file = os.path.join(eval_dir, 'summary_report.txt')
    else:
        out_file = os.path.join(eval_dir, 'circle_summary_report.txt')

    with open(out_file, "w") as text_file:

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Summary of geometric evaluation measures. \n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV', 'Myo']:

            text_file.write(struc_name)
            text_file.write('\n')

            for cardiac_phase in ['ED', 'ES']:

                text_file.write('    {}\n'.format(cardiac_phase))

                dat = df.loc[(df['phase'] == cardiac_phase) & (df['struc'] == struc_name)]

                for measure_name in ['dice', 'hd']:

                    text_file.write('       {} -- mean (std): {:.3f} ({:.3f}) \n'.format(measure_name,
                                                                                         np.mean(dat[measure_name]),
                                                                                         np.std(dat[measure_name])))

                    ind_med = np.argsort(dat[measure_name]).iloc[len(dat[measure_name]) // 2]
                    text_file.write('             median {}: {:.3f} ({})\n'.format(measure_name,
                                                                                   dat[measure_name].iloc[ind_med],
                                                                                   dat['filename'].iloc[ind_med]))

                    if measure_name == 'dice':
                        ind_worst = np.argsort(dat[measure_name]).iloc[0]
                        text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                                      dat[measure_name].iloc[ind_worst],
                                                                                      dat['filename'].iloc[ind_worst]))

                        ind_best = np.argsort(dat[measure_name]).iloc[-1]
                        text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                                     dat[measure_name].iloc[ind_best],
                                                                                     dat['filename'].iloc[ind_best]))
                    else:
                        ind_worst = np.argsort(dat[measure_name]).iloc[-1]
                        text_file.write('             worst {}: {:.3f} ({})\n'.format(measure_name,
                                                                                      dat[measure_name].iloc[ind_worst],
                                                                                      dat['filename'].iloc[ind_worst]))

                        ind_best = np.argsort(dat[measure_name]).iloc[0]
                        text_file.write('             best {}: {:.3f} ({})\n'.format(measure_name,
                                                                                     dat[measure_name].iloc[ind_best],
                                                                                     dat['filename'].iloc[ind_best]))

        text_file.write('\n\n-------------------------------------------------------------------------------------\n')
        text_file.write('Correlation between prediction and ground truth\n')
        text_file.write('-------------------------------------------------------------------------------------\n\n')

        for struc_name in ['LV', 'RV']:
            lv = df.loc[df['struc'] == struc_name]

            ED_vol = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
            ES_vol = np.array(lv.loc[(lv['phase'] == 'ES')]['vol'])
            SV_pred = ED_vol - ES_vol
            EF_pred = SV_pred / ED_vol

            ED_vol_gt = np.array(lv.loc[lv['phase'] == 'ED']['vol_gt'])
            ES_vol_gt = np.array(lv.loc[(lv['phase'] == 'ES')]['vol_gt'])
            SV_gt = ED_vol_gt - ES_vol_gt
            EF_gt = SV_gt / ED_vol_gt

            #correlation
            # EF_corr, _ = stats.pearsonr(EF_pred, EF_gt)
            # EF_corr = (np.cov(EF_pred, EF_gt)[1, 0] / (np.std(EF_pred) * np.std(EF_gt)))
            EDV_corr = stats.pearsonr(ED_vol, ED_vol_gt)
            ESV_corr = stats.pearsonr(ES_vol, ES_vol_gt)
            EF_corr = stats.pearsonr(EF_pred, EF_gt)
            SV_corr = stats.pearsonr(SV_pred, SV_gt)
            text_file.write('{}, EDV corr: {}\n\n'.format(struc_name, EDV_corr[0] * 100))
            text_file.write('{}, ESV corr: {}\n\n'.format(struc_name, ESV_corr[0] * 100))
            text_file.write('{}, SV corr: {}\n\n'.format(struc_name, SV_corr[0] * 100))
            text_file.write('{}, EF corr: {}\n\n'.format(struc_name, EF_corr[0] * 100))

            #Bland-Altman
            bias_EDV = np.mean(ED_vol_gt - ED_vol)
            bias_ESV = np.mean(ES_vol_gt - ES_vol)
            bias_SV = np.mean(SV_gt - SV_pred)
            bias_EF = np.mean(EF_gt - EF_pred)

            LOA_EDV = 1.96*np.std(ED_vol_gt - ED_vol)
            LOA_ESV = 1.96*np.std(ES_vol_gt - ES_vol)
            LOA_SV = 1.96*np.std(SV_gt - SV_pred)
            LOA_EF = 1.96*np.std(EF_gt - EF_pred)

            _, pvalueED = stats.ttest_1samp((ED_vol_gt - ED_vol), popmean=np.mean(ED_vol_gt - ED_vol))
            _, pvalueES = stats.ttest_1samp((ES_vol_gt - ES_vol), popmean=np.mean(ES_vol_gt - ES_vol))
            _, pvalueSV = stats.ttest_1samp((SV_gt - SV_pred), popmean=np.mean(SV_gt - SV_pred))
            _, pvalueEF = stats.ttest_1samp((EF_gt - EF_pred), popmean=np.mean(EF_gt - EF_pred))

            text_file.write('{}, EDV bias: {}, LOA: {}, pvalue: {}\n\n'.format(struc_name, bias_EDV, LOA_EDV, pvalueED))
            text_file.write('{}, ESV bias: {}, LOA: {}, pvalue: {}\n\n'.format(struc_name, bias_ESV, LOA_ESV, pvalueES))
            text_file.write('{}, SV bias: {}, LOA: {}, pvalue: {}\n\n'.format(struc_name, bias_SV, LOA_SV, pvalueSV))
            text_file.write('{}, EF bias: {}, LOA: {}, pvalue: {}\n\n'.format(struc_name, bias_EF, LOA_EF, pvalueEF))


        for struc_name in ['Myo']:
            lv = df.loc[df['struc'] == struc_name]
            MYmassED = np.array(lv.loc[lv['phase'] == 'ED']['vol'])
            MYmassES = np.array(lv.loc[lv['phase'] == 'ES']['vol'])
            MYmassED_gt = np.array(lv.loc[lv['phase'] == 'ED']['vol_gt'])
            MYmassES_gt = np.array(lv.loc[lv['phase'] == 'ES']['vol_gt'])
            
            MYED_corr = stats.pearsonr(MYmassED, MYmassED_gt)
            MYES_corr = stats.pearsonr(MYmassES, MYmassES_gt)

            text_file.write('MYmass ED corr: {}\n\n'.format(MYED_corr[0] * 100))
            text_file.write('MYmass ES corr: {}\n\n'.format(MYES_corr[0] * 100))

            bias_MYmassED = np.mean(MYmassED_gt - MYmassED)
            LOA_MYmassED = 1.96*np.std(MYmassED_gt - MYmassED)
            text_file.write('MYmass ED bias: {}, LOA: {}\n\n'.format(bias_MYmassED, LOA_MYmassED))
            
            bias_MYmassES = np.mean(MYmassES_gt - MYmassES)
            LOA_MYmassES = 1.96*np.std(MYmassES_gt - MYmassES)
            text_file.write('MYmass ES bias: {}, LOA: {}\n\n'.format(bias_MYmassES, LOA_MYmassES))


def boxplot_metrics(df, eval_dir, circle=False):
    """
    Create summary boxplots of all geometric measures.
    :param df:
    :param eval_dir:
    :param circle: 
    :return:
    """
    if not circle:
        boxplots_file = os.path.join(eval_dir, 'boxplots.png')
    else:
        boxplots_file = os.path.join(eval_dir, 'circle_boxplots.png')

    fig, axes = plt.subplots(2, 1)
    fig.set_figheight(14)
    fig.set_figwidth(7)

    sns.boxplot(x='struc', y='dice', hue='phase', data=df, palette="PRGn", ax=axes[0])
    sns.boxplot(x='struc', y='hd', hue='phase', data=df, palette="PRGn", ax=axes[1])

    plt.savefig(boxplots_file)
    plt.close()

    return 0


def main(path_pred):
    logging.info(path_pred)

    if os.path.exists(os.path.join(path_pred, 'pred_on_dice.hdf5')):
        path_eval = os.path.join(path_pred, 'eval_on_dice')
        if not os.path.exists(path_eval):
            utils.makefolder(path_eval)
            logging.info(path_eval)
            df1, df2 = compute_metrics_on_directories_raw(path_pred, path_eval, dice=True)

            df1.to_pickle(os.path.join(path_eval, 'df1_dice.pkl'))
            df2.to_pickle(os.path.join(path_eval, 'df2_dice.pkl'))
            df1.to_excel(os.path.join(path_eval, 'Excel_df1_dice.xlsx'))
            df2.to_excel(os.path.join(path_eval, 'Excel_df2_dice.xlsx'))
            print_stats(df1, path_eval, circle=False)
            print_latex_tables(df1, path_eval, circle=False)
            boxplot_metrics(df1, path_eval, circle=False)
            print_stats(df2, path_eval, circle=True)
            print_latex_tables(df2, path_eval, circle=True)
            boxplot_metrics(df2, path_eval, circle=True)

            logging.info('------------Average Dice Figures----------')
            logging.info('Dice 1: %f' % np.mean(df1.loc[df1['struc'] == 'LV']['dice']))
            logging.info('Dice 2: %f' % np.mean(df1.loc[df1['struc'] == 'RV']['dice']))
            logging.info('Dice 3: %f' % np.mean(df1.loc[df1['struc'] == 'Myo']['dice']))
            logging.info('Mean dice: %f' % np.mean(np.mean(df1['dice'])))
            logging.info('------------------------------------------')

    if os.path.exists(os.path.join(path_pred, 'pred_on_loss.hdf5')):
        path_eval = os.path.join(path_pred, 'eval_on_loss')
        if not os.path.exists(path_eval):
            utils.makefolder(path_eval)
            logging.info(path_eval)
            df1, df2 = compute_metrics_on_directories_raw(path_pred, path_eval, dice=False)

            df1.to_pickle(os.path.join(path_eval, 'df1_loss.pkl'))
            df2.to_pickle(os.path.join(path_eval, 'df2_loss.pkl'))
            df1.to_excel(os.path.join(path_eval, 'Excel_df1_loss.xlsx'))
            df2.to_excel(os.path.join(path_eval, 'Excel_df2_loss.xlsx'))
            print_stats(df1, path_eval, circle=False)
            print_latex_tables(df1, path_eval, circle=False)
            boxplot_metrics(df1, path_eval, circle=False)
            print_stats(df2, path_eval, circle=True)
            print_latex_tables(df2, path_eval, circle=True)
            boxplot_metrics(df2, path_eval, circle=True)

            logging.info('------------Average Dice Figures----------')
            logging.info('Dice 1: %f' % np.mean(df1.loc[df1['struc'] == 'LV']['dice']))
            logging.info('Dice 2: %f' % np.mean(df1.loc[df1['struc'] == 'RV']['dice']))
            logging.info('Dice 3: %f' % np.mean(df1.loc[df1['struc'] == 'Myo']['dice']))
            logging.info('Mean dice: %f' % np.mean(np.mean(df1['dice'])))
            logging.info('------------------------------------------')
