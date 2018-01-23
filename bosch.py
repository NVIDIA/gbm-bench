# source: https://www.kaggle.com/c/bosch-production-line-performance

from __future__ import print_function

import os
import time
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *


def prepareImpl(dbFolder, testSize, shuffle):
    # reading compressed csv is supported in pandas
    csv_name = 'train_numeric.zip'
    cols = [
        'Id',
        'L0_S0_F0', 'L0_S0_F2', 'L0_S0_F4', 'L0_S0_F6', 'L0_S0_F8', 'L0_S0_F10',
        'L0_S0_F12', 'L0_S0_F14', 'L0_S0_F16', 'L0_S0_F18', 'L0_S0_F20', 'L0_S0_F22',
        'L0_S1_F24', 'L0_S1_F28', 'L0_S2_F32', 'L0_S2_F36', 'L0_S2_F40', 'L0_S2_F44',
        'L0_S2_F48', 'L0_S2_F52', 'L0_S2_F56', 'L0_S2_F60', 'L0_S2_F64', 'L0_S3_F68',
        'L0_S3_F72', 'L0_S3_F76', 'L0_S3_F80', 'L0_S3_F84', 'L0_S3_F88', 'L0_S3_F92',
        'L0_S3_F96', 'L0_S3_F100', 'L0_S4_F104', 'L0_S4_F109', 'L0_S5_F114', 'L0_S5_F116',
        'L0_S6_F118', 'L0_S6_F122', 'L0_S6_F132', 'L0_S7_F136', 'L0_S7_F138',
        'L0_S7_F142', 'L0_S8_F144', 'L0_S8_F146', 'L0_S8_F149', 'L0_S9_F155',
        'L0_S9_F160', 'L0_S9_F165', 'L0_S9_F170', 'L0_S9_F175', 'L0_S9_F180',
        'L0_S9_F185', 'L0_S9_F190', 'L0_S9_F195', 'L0_S9_F200', 'L0_S9_F205',
        'L0_S9_F210', 'L0_S10_F219', 'L0_S10_F224', 'L0_S10_F229', 'L0_S10_F234',
        'L0_S10_F239', 'L0_S10_F244', 'L0_S10_F249', 'L0_S10_F254', 'L0_S10_F259',
        'L0_S10_F264', 'L0_S10_F269', 'L0_S10_F274', 'L0_S11_F282', 'L0_S11_F286',
        'L0_S11_F290', 'L0_S11_F294', 'L0_S11_F298', 'L0_S11_F302', 'L0_S11_F306',
        'L0_S11_F310', 'L0_S11_F314', 'L0_S11_F318', 'L0_S11_F322', 'L0_S11_F326',
        'L0_S12_F330', 'L0_S12_F332', 'L0_S12_F334', 'L0_S12_F336', 'L0_S12_F338',
        'L0_S12_F340', 'L0_S12_F342', 'L0_S12_F344', 'L0_S12_F346', 'L0_S12_F348',
        'L0_S12_F350', 'L0_S12_F352', 'L0_S13_F354', 'L0_S13_F356', 'L0_S14_F358',
        'L0_S14_F362', 'L0_S14_F366', 'L0_S14_F370', 'L0_S14_F374', 'L0_S14_F378',
        'L0_S14_F382', 'L0_S14_F386', 'L0_S14_F390', 'L0_S15_F394', 'L0_S15_F397',
        'L0_S15_F400', 'L0_S15_F403', 'L0_S15_F406', 'L0_S15_F409', 'L0_S15_F412',
        'L0_S15_F415', 'L0_S15_F418', 'L0_S16_F421', 'L0_S16_F426', 'L0_S17_F431',
        'L0_S17_F433', 'L0_S18_F435', 'L0_S18_F439', 'L0_S18_F449', 'L0_S19_F453',
        'L0_S19_F455', 'L0_S19_F459', 'L0_S20_F461', 'L0_S20_F463', 'L0_S20_F466',
        'L0_S21_F472', 'L0_S21_F477', 'L0_S21_F482', 'L0_S21_F487', 'L0_S21_F492',
        'L0_S21_F497', 'L0_S21_F502', 'L0_S21_F507', 'L0_S21_F512', 'L0_S21_F517',
        'L0_S21_F522', 'L0_S21_F527', 'L0_S21_F532', 'L0_S21_F537', 'L0_S22_F546',
        'L0_S22_F551', 'L0_S22_F556', 'L0_S22_F561', 'L0_S22_F566', 'L0_S22_F571',
        'L0_S22_F576', 'L0_S22_F581', 'L0_S22_F586', 'L0_S22_F591', 'L0_S22_F596',
        'L0_S22_F601', 'L0_S22_F606', 'L0_S22_F611', 'L0_S23_F619', 'L0_S23_F623',
        'L0_S23_F627', 'L0_S23_F631', 'L0_S23_F635', 'L0_S23_F639', 'L0_S23_F643',
        'L0_S23_F647', 'L0_S23_F651', 'L0_S23_F655', 'L0_S23_F659', 'L0_S23_F663',
        'L0_S23_F667', 'L0_S23_F671', 'L1_S24_F679', 'L1_S24_F683', 'L1_S24_F687',
        'L1_S24_F691', 'L1_S24_F700', 'L1_S24_F719', 'L1_S24_F728', 'L1_S24_F733',
        'L1_S24_F746', 'L1_S24_F751', 'L1_S24_F756', 'L1_S24_F761', 'L1_S24_F766',
        'L1_S24_F775', 'L1_S24_F780', 'L1_S24_F785', 'L1_S24_F790', 'L1_S24_F795',
        'L1_S24_F800', 'L1_S24_F802', 'L1_S24_F806', 'L1_S24_F808', 'L1_S24_F810',
        'L1_S24_F812', 'L1_S24_F814', 'L1_S24_F816', 'L1_S24_F829', 'L1_S24_F834',
        'L1_S24_F839', 'L1_S24_F844', 'L1_S24_F857', 'L1_S24_F862', 'L1_S24_F867',
        'L1_S24_F872', 'L1_S24_F877', 'L1_S24_F882', 'L1_S24_F887', 'L1_S24_F892',
        'L1_S24_F897', 'L1_S24_F902', 'L1_S24_F907', 'L1_S24_F920', 'L1_S24_F925',
        'L1_S24_F930', 'L1_S24_F935', 'L1_S24_F948', 'L1_S24_F953', 'L1_S24_F958',
        'L1_S24_F963', 'L1_S24_F968', 'L1_S24_F973', 'L1_S24_F978', 'L1_S24_F983',
        'L1_S24_F988', 'L1_S24_F993', 'L1_S24_F998', 'L1_S24_F1000', 'L1_S24_F1002',
        'L1_S24_F1004', 'L1_S24_F1006', 'L1_S24_F1008', 'L1_S24_F1010', 'L1_S24_F1012',
        'L1_S24_F1014', 'L1_S24_F1016', 'L1_S24_F1021', 'L1_S24_F1026', 'L1_S24_F1031',
        'L1_S24_F1036', 'L1_S24_F1041', 'L1_S24_F1046', 'L1_S24_F1051', 'L1_S24_F1056',
        'L1_S24_F1068', 'L1_S24_F1072', 'L1_S24_F1079', 'L1_S24_F1083', 'L1_S24_F1087',
        'L1_S24_F1094', 'L1_S24_F1098', 'L1_S24_F1102', 'L1_S24_F1106', 'L1_S24_F1110',
        'L1_S24_F1118', 'L1_S24_F1122', 'L1_S24_F1126', 'L1_S24_F1130', 'L1_S24_F1134',
        'L1_S24_F1145', 'L1_S24_F1148', 'L1_S24_F1161', 'L1_S24_F1166', 'L1_S24_F1170',
        'L1_S24_F1172', 'L1_S24_F1174', 'L1_S24_F1176', 'L1_S24_F1180', 'L1_S24_F1184',
        'L1_S24_F1197', 'L1_S24_F1202', 'L1_S24_F1207', 'L1_S24_F1212', 'L1_S24_F1225',
        'L1_S24_F1230', 'L1_S24_F1235', 'L1_S24_F1240', 'L1_S24_F1245', 'L1_S24_F1250',
        'L1_S24_F1255', 'L1_S24_F1260', 'L1_S24_F1265', 'L1_S24_F1270', 'L1_S24_F1275',
        'L1_S24_F1288', 'L1_S24_F1293', 'L1_S24_F1298', 'L1_S24_F1303', 'L1_S24_F1316',
        'L1_S24_F1321', 'L1_S24_F1326', 'L1_S24_F1331', 'L1_S24_F1336', 'L1_S24_F1341',
        'L1_S24_F1346', 'L1_S24_F1351', 'L1_S24_F1356', 'L1_S24_F1361', 'L1_S24_F1366',
        'L1_S24_F1371', 'L1_S24_F1376', 'L1_S24_F1381', 'L1_S24_F1386', 'L1_S24_F1391',
        'L1_S24_F1396', 'L1_S24_F1401', 'L1_S24_F1406', 'L1_S24_F1411', 'L1_S24_F1416',
        'L1_S24_F1421', 'L1_S24_F1426', 'L1_S24_F1431', 'L1_S24_F1436', 'L1_S24_F1441',
        'L1_S24_F1446', 'L1_S24_F1451', 'L1_S24_F1463', 'L1_S24_F1467', 'L1_S24_F1474',
        'L1_S24_F1478', 'L1_S24_F1482', 'L1_S24_F1486', 'L1_S24_F1490', 'L1_S24_F1494',
        'L1_S24_F1498', 'L1_S24_F1502', 'L1_S24_F1506', 'L1_S24_F1512', 'L1_S24_F1514',
        'L1_S24_F1516', 'L1_S24_F1518', 'L1_S24_F1520', 'L1_S24_F1539', 'L1_S24_F1544',
        'L1_S24_F1565', 'L1_S24_F1567', 'L1_S24_F1569', 'L1_S24_F1571', 'L1_S24_F1573',
        'L1_S24_F1575', 'L1_S24_F1578', 'L1_S24_F1581', 'L1_S24_F1594', 'L1_S24_F1599',
        'L1_S24_F1604', 'L1_S24_F1609', 'L1_S24_F1622', 'L1_S24_F1627', 'L1_S24_F1632',
        'L1_S24_F1637', 'L1_S24_F1642', 'L1_S24_F1647', 'L1_S24_F1652', 'L1_S24_F1657',
        'L1_S24_F1662', 'L1_S24_F1667', 'L1_S24_F1672', 'L1_S24_F1685', 'L1_S24_F1690',
        'L1_S24_F1695', 'L1_S24_F1700', 'L1_S24_F1713', 'L1_S24_F1718', 'L1_S24_F1723',
        'L1_S24_F1728', 'L1_S24_F1733', 'L1_S24_F1738', 'L1_S24_F1743', 'L1_S24_F1748',
        'L1_S24_F1753', 'L1_S24_F1758', 'L1_S24_F1763', 'L1_S24_F1768', 'L1_S24_F1773',
        'L1_S24_F1778', 'L1_S24_F1783', 'L1_S24_F1788', 'L1_S24_F1793', 'L1_S24_F1798',
        'L1_S24_F1803', 'L1_S24_F1808', 'L1_S24_F1810', 'L1_S24_F1812', 'L1_S24_F1814',
        'L1_S24_F1816', 'L1_S24_F1818', 'L1_S24_F1820', 'L1_S24_F1822', 'L1_S24_F1824',
        'L1_S24_F1829', 'L1_S24_F1831', 'L1_S24_F1834', 'L1_S24_F1836', 'L1_S24_F1838',
        'L1_S24_F1840', 'L1_S24_F1842', 'L1_S24_F1844', 'L1_S24_F1846', 'L1_S24_F1848',
        'L1_S24_F1850', 'L1_S25_F1855', 'L1_S25_F1858', 'L1_S25_F1865', 'L1_S25_F1869',
        'L1_S25_F1873', 'L1_S25_F1877', 'L1_S25_F1881', 'L1_S25_F1885', 'L1_S25_F1890',
        'L1_S25_F1892', 'L1_S25_F1894', 'L1_S25_F1896', 'L1_S25_F1900', 'L1_S25_F1909',
        'L1_S25_F1914', 'L1_S25_F1919', 'L1_S25_F1924', 'L1_S25_F1929', 'L1_S25_F1938',
        'L1_S25_F1943', 'L1_S25_F1948', 'L1_S25_F1953', 'L1_S25_F1958', 'L1_S25_F1963',
        'L1_S25_F1968', 'L1_S25_F1973', 'L1_S25_F1978', 'L1_S25_F1987', 'L1_S25_F1992',
        'L1_S25_F1997', 'L1_S25_F2002', 'L1_S25_F2007', 'L1_S25_F2016', 'L1_S25_F2021',
        'L1_S25_F2026', 'L1_S25_F2031', 'L1_S25_F2036', 'L1_S25_F2041', 'L1_S25_F2046',
        'L1_S25_F2051', 'L1_S25_F2056', 'L1_S25_F2061', 'L1_S25_F2066', 'L1_S25_F2071',
        'L1_S25_F2076', 'L1_S25_F2081', 'L1_S25_F2086', 'L1_S25_F2091', 'L1_S25_F2096',
        'L1_S25_F2101', 'L1_S25_F2106', 'L1_S25_F2111', 'L1_S25_F2116', 'L1_S25_F2121',
        'L1_S25_F2126', 'L1_S25_F2131', 'L1_S25_F2136', 'L1_S25_F2144', 'L1_S25_F2147',
        'L1_S25_F2152', 'L1_S25_F2155', 'L1_S25_F2158', 'L1_S25_F2161', 'L1_S25_F2164',
        'L1_S25_F2167', 'L1_S25_F2170', 'L1_S25_F2173', 'L1_S25_F2176', 'L1_S25_F2181',
        'L1_S25_F2184', 'L1_S25_F2187', 'L1_S25_F2190', 'L1_S25_F2193', 'L1_S25_F2196',
        'L1_S25_F2199', 'L1_S25_F2202', 'L1_S25_F2207', 'L1_S25_F2210', 'L1_S25_F2217',
        'L1_S25_F2220', 'L1_S25_F2223', 'L1_S25_F2226', 'L1_S25_F2231', 'L1_S25_F2233',
        'L1_S25_F2237', 'L1_S25_F2239', 'L1_S25_F2241', 'L1_S25_F2243', 'L1_S25_F2245',
        'L1_S25_F2247', 'L1_S25_F2249', 'L1_S25_F2258', 'L1_S25_F2263', 'L1_S25_F2268',
        'L1_S25_F2273', 'L1_S25_F2278', 'L1_S25_F2287', 'L1_S25_F2292', 'L1_S25_F2297',
        'L1_S25_F2302', 'L1_S25_F2307', 'L1_S25_F2312', 'L1_S25_F2317', 'L1_S25_F2322',
        'L1_S25_F2327', 'L1_S25_F2336', 'L1_S25_F2341', 'L1_S25_F2346', 'L1_S25_F2351',
        'L1_S25_F2356', 'L1_S25_F2365', 'L1_S25_F2370', 'L1_S25_F2375', 'L1_S25_F2380',
        'L1_S25_F2385', 'L1_S25_F2390', 'L1_S25_F2395', 'L1_S25_F2400', 'L1_S25_F2405',
        'L1_S25_F2408', 'L1_S25_F2411', 'L1_S25_F2414', 'L1_S25_F2417', 'L1_S25_F2420',
        'L1_S25_F2423', 'L1_S25_F2426', 'L1_S25_F2429', 'L1_S25_F2431', 'L1_S25_F2433',
        'L1_S25_F2435', 'L1_S25_F2437', 'L1_S25_F2439', 'L1_S25_F2441', 'L1_S25_F2443',
        'L1_S25_F2449', 'L1_S25_F2451', 'L1_S25_F2454', 'L1_S25_F2456', 'L1_S25_F2458',
        'L1_S25_F2460', 'L1_S25_F2462', 'L1_S25_F2464', 'L1_S25_F2466', 'L1_S25_F2468',
        'L1_S25_F2472', 'L1_S25_F2475', 'L1_S25_F2478', 'L1_S25_F2481', 'L1_S25_F2484',
        'L1_S25_F2487', 'L1_S25_F2490', 'L1_S25_F2493', 'L1_S25_F2498', 'L1_S25_F2500',
        'L1_S25_F2504', 'L1_S25_F2506', 'L1_S25_F2508', 'L1_S25_F2510', 'L1_S25_F2512',
        'L1_S25_F2514', 'L1_S25_F2516', 'L1_S25_F2525', 'L1_S25_F2530', 'L1_S25_F2535',
        'L1_S25_F2540', 'L1_S25_F2545', 'L1_S25_F2554', 'L1_S25_F2559', 'L1_S25_F2564',
        'L1_S25_F2569', 'L1_S25_F2574', 'L1_S25_F2579', 'L1_S25_F2584', 'L1_S25_F2589',
        'L1_S25_F2594', 'L1_S25_F2603', 'L1_S25_F2608', 'L1_S25_F2613', 'L1_S25_F2618',
        'L1_S25_F2623', 'L1_S25_F2632', 'L1_S25_F2637', 'L1_S25_F2642', 'L1_S25_F2647',
        'L1_S25_F2652', 'L1_S25_F2657', 'L1_S25_F2662', 'L1_S25_F2667', 'L1_S25_F2672',
        'L1_S25_F2677', 'L1_S25_F2682', 'L1_S25_F2687', 'L1_S25_F2692', 'L1_S25_F2697',
        'L1_S25_F2702', 'L1_S25_F2707', 'L1_S25_F2712', 'L1_S25_F2714', 'L1_S25_F2716',
        'L1_S25_F2718', 'L1_S25_F2720', 'L1_S25_F2722', 'L1_S25_F2724', 'L1_S25_F2726',
        'L1_S25_F2732', 'L1_S25_F2734', 'L1_S25_F2737', 'L1_S25_F2739', 'L1_S25_F2741',
        'L1_S25_F2743', 'L1_S25_F2745', 'L1_S25_F2747', 'L1_S25_F2749', 'L1_S25_F2751',
        'L1_S25_F2755', 'L1_S25_F2758', 'L1_S25_F2761', 'L1_S25_F2764', 'L1_S25_F2767',
        'L1_S25_F2770', 'L1_S25_F2773', 'L1_S25_F2776', 'L1_S25_F2781', 'L1_S25_F2783',
        'L1_S25_F2787', 'L1_S25_F2789', 'L1_S25_F2791', 'L1_S25_F2793', 'L1_S25_F2795',
        'L1_S25_F2797', 'L1_S25_F2799', 'L1_S25_F2808', 'L1_S25_F2813', 'L1_S25_F2818',
        'L1_S25_F2823', 'L1_S25_F2828', 'L1_S25_F2837', 'L1_S25_F2842', 'L1_S25_F2847',
        'L1_S25_F2852', 'L1_S25_F2857', 'L1_S25_F2862', 'L1_S25_F2867', 'L1_S25_F2872',
        'L1_S25_F2877', 'L1_S25_F2886', 'L1_S25_F2891', 'L1_S25_F2896', 'L1_S25_F2901',
        'L1_S25_F2906', 'L1_S25_F2915', 'L1_S25_F2920', 'L1_S25_F2925', 'L1_S25_F2930',
        'L1_S25_F2935', 'L1_S25_F2940', 'L1_S25_F2945', 'L1_S25_F2950', 'L1_S25_F2955',
        'L1_S25_F2960', 'L1_S25_F2965', 'L1_S25_F2970', 'L1_S25_F2975', 'L1_S25_F2980',
        'L1_S25_F2985', 'L1_S25_F2990', 'L1_S25_F2995', 'L1_S25_F2997', 'L1_S25_F2999',
        'L1_S25_F3001', 'L1_S25_F3003', 'L1_S25_F3005', 'L1_S25_F3007', 'L1_S25_F3009',
        'L1_S25_F3015', 'L1_S25_F3017', 'L1_S25_F3020', 'L1_S25_F3022', 'L1_S25_F3024',
        'L1_S25_F3026', 'L1_S25_F3028', 'L1_S25_F3030', 'L1_S25_F3032', 'L1_S25_F3034',
        'L2_S26_F3036', 'L2_S26_F3040', 'L2_S26_F3047', 'L2_S26_F3051', 'L2_S26_F3055',
        'L2_S26_F3062', 'L2_S26_F3069', 'L2_S26_F3073', 'L2_S26_F3077', 'L2_S26_F3106',
        'L2_S26_F3113', 'L2_S26_F3117', 'L2_S26_F3121', 'L2_S26_F3125', 'L2_S27_F3129',
        'L2_S27_F3133', 'L2_S27_F3140', 'L2_S27_F3144', 'L2_S27_F3148', 'L2_S27_F3155',
        'L2_S27_F3162', 'L2_S27_F3166', 'L2_S27_F3170', 'L2_S27_F3199', 'L2_S27_F3206',
        'L2_S27_F3210', 'L2_S27_F3214', 'L2_S27_F3218', 'L2_S28_F3222', 'L2_S28_F3226',
        'L2_S28_F3233', 'L2_S28_F3237', 'L2_S28_F3241', 'L2_S28_F3248', 'L2_S28_F3255',
        'L2_S28_F3259', 'L2_S28_F3263', 'L2_S28_F3292', 'L2_S28_F3299', 'L2_S28_F3303',
        'L2_S28_F3307', 'L2_S28_F3311', 'L3_S29_F3315', 'L3_S29_F3318', 'L3_S29_F3321',
        'L3_S29_F3324', 'L3_S29_F3327', 'L3_S29_F3330', 'L3_S29_F3333', 'L3_S29_F3336',
        'L3_S29_F3339', 'L3_S29_F3342', 'L3_S29_F3345', 'L3_S29_F3348', 'L3_S29_F3351',
        'L3_S29_F3354', 'L3_S29_F3357', 'L3_S29_F3360', 'L3_S29_F3367', 'L3_S29_F3370',
        'L3_S29_F3373', 'L3_S29_F3376', 'L3_S29_F3379', 'L3_S29_F3382', 'L3_S29_F3385',
        'L3_S29_F3388', 'L3_S29_F3395', 'L3_S29_F3398', 'L3_S29_F3401', 'L3_S29_F3404',
        'L3_S29_F3407', 'L3_S29_F3412', 'L3_S29_F3421', 'L3_S29_F3424', 'L3_S29_F3427',
        'L3_S29_F3430', 'L3_S29_F3433', 'L3_S29_F3436', 'L3_S29_F3439', 'L3_S29_F3442',
        'L3_S29_F3449', 'L3_S29_F3452', 'L3_S29_F3455', 'L3_S29_F3458', 'L3_S29_F3461',
        'L3_S29_F3464', 'L3_S29_F3467', 'L3_S29_F3470', 'L3_S29_F3473', 'L3_S29_F3476',
        'L3_S29_F3479', 'L3_S29_F3482', 'L3_S29_F3485', 'L3_S29_F3488', 'L3_S29_F3491',
        'L3_S30_F3494', 'L3_S30_F3499', 'L3_S30_F3504', 'L3_S30_F3509', 'L3_S30_F3514',
        'L3_S30_F3519', 'L3_S30_F3524', 'L3_S30_F3529', 'L3_S30_F3534', 'L3_S30_F3539',
        'L3_S30_F3544', 'L3_S30_F3549', 'L3_S30_F3554', 'L3_S30_F3559', 'L3_S30_F3564',
        'L3_S30_F3569', 'L3_S30_F3574', 'L3_S30_F3579', 'L3_S30_F3584', 'L3_S30_F3589',
        'L3_S30_F3594', 'L3_S30_F3599', 'L3_S30_F3604', 'L3_S30_F3609', 'L3_S30_F3614',
        'L3_S30_F3619', 'L3_S30_F3624', 'L3_S30_F3629', 'L3_S30_F3634', 'L3_S30_F3639',
        'L3_S30_F3644', 'L3_S30_F3649', 'L3_S30_F3654', 'L3_S30_F3659', 'L3_S30_F3664',
        'L3_S30_F3669', 'L3_S30_F3674', 'L3_S30_F3679', 'L3_S30_F3684', 'L3_S30_F3689',
        'L3_S30_F3694', 'L3_S30_F3699', 'L3_S30_F3704', 'L3_S30_F3709', 'L3_S30_F3714',
        'L3_S30_F3719', 'L3_S30_F3724', 'L3_S30_F3729', 'L3_S30_F3734', 'L3_S30_F3739',
        'L3_S30_F3744', 'L3_S30_F3749', 'L3_S30_F3754', 'L3_S30_F3759', 'L3_S30_F3764',
        'L3_S30_F3769', 'L3_S30_F3774', 'L3_S30_F3779', 'L3_S30_F3784', 'L3_S30_F3789',
        'L3_S30_F3794', 'L3_S30_F3799', 'L3_S30_F3804', 'L3_S30_F3809', 'L3_S30_F3814',
        'L3_S30_F3819', 'L3_S30_F3824', 'L3_S30_F3829', 'L3_S31_F3834', 'L3_S31_F3838',
        'L3_S31_F3842', 'L3_S31_F3846', 'L3_S32_F3850', 'L3_S33_F3855', 'L3_S33_F3857',
        'L3_S33_F3859', 'L3_S33_F3861', 'L3_S33_F3863', 'L3_S33_F3865', 'L3_S33_F3867',
        'L3_S33_F3869', 'L3_S33_F3871', 'L3_S33_F3873', 'L3_S34_F3876', 'L3_S34_F3878',
        'L3_S34_F3880', 'L3_S34_F3882', 'L3_S35_F3884', 'L3_S35_F3889', 'L3_S35_F3894',
        'L3_S35_F3896', 'L3_S35_F3898', 'L3_S35_F3903', 'L3_S35_F3908', 'L3_S35_F3913',
        'L3_S36_F3918', 'L3_S36_F3920', 'L3_S36_F3922', 'L3_S36_F3924', 'L3_S36_F3926',
        'L3_S36_F3930', 'L3_S36_F3934', 'L3_S36_F3938', 'L3_S37_F3944', 'L3_S37_F3946',
        'L3_S37_F3948', 'L3_S37_F3950', 'L3_S38_F3952', 'L3_S38_F3956', 'L3_S38_F3960',
        'L3_S39_F3964', 'L3_S39_F3968', 'L3_S39_F3972', 'L3_S39_F3976', 'L3_S40_F3980',
        'L3_S40_F3982', 'L3_S40_F3984', 'L3_S40_F3986', 'L3_S40_F3988', 'L3_S40_F3990',
        'L3_S40_F3992', 'L3_S40_F3994', 'L3_S41_F3996', 'L3_S41_F3998', 'L3_S41_F4000',
        'L3_S41_F4002', 'L3_S41_F4004', 'L3_S41_F4006', 'L3_S41_F4008', 'L3_S41_F4011',
        'L3_S41_F4014', 'L3_S41_F4016', 'L3_S41_F4018', 'L3_S41_F4020', 'L3_S41_F4023',
        'L3_S41_F4026', 'L3_S43_F4060', 'L3_S43_F4065', 'L3_S43_F4070', 'L3_S43_F4075',
        'L3_S43_F4080', 'L3_S43_F4085', 'L3_S43_F4090', 'L3_S43_F4095', 'L3_S44_F4100',
        'L3_S44_F4103', 'L3_S44_F4106', 'L3_S44_F4109', 'L3_S44_F4112', 'L3_S44_F4115',
        'L3_S44_F4118', 'L3_S44_F4121', 'L3_S45_F4124', 'L3_S45_F4126', 'L3_S45_F4128',
        'L3_S45_F4130', 'L3_S45_F4132', 'L3_S47_F4138', 'L3_S47_F4143', 'L3_S47_F4148',
        'L3_S47_F4153', 'L3_S47_F4158', 'L3_S47_F4163', 'L3_S47_F4168', 'L3_S47_F4173',
        'L3_S47_F4178', 'L3_S47_F4183', 'L3_S47_F4188', 'L3_S48_F4193', 'L3_S48_F4196',
        'L3_S48_F4198', 'L3_S48_F4200', 'L3_S48_F4202', 'L3_S48_F4204', 'L3_S49_F4206',
        'L3_S49_F4211', 'L3_S49_F4216', 'L3_S49_F4221', 'L3_S49_F4226', 'L3_S49_F4231',
        'L3_S49_F4236', 'L3_S50_F4241', 'L3_S50_F4243', 'L3_S50_F4245', 'L3_S50_F4247',
        'L3_S50_F4249', 'L3_S50_F4251', 'L3_S50_F4253', 'L3_S51_F4256', 'L3_S51_F4258',
        'L3_S51_F4260', 'L3_S51_F4262',
        'Response'
    ]
    csv_file = os.path.join(dbFolder, csv_name)
    start = time.time()
    # to avoid the costly duplication process, directly work on the read pandas frame
    X = pd.read_csv(csv_file, compression='zip', dtype=np.float32)
    del X['Id']
    y = X['Response']
    del X['Response']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        shuffle=shuffle,
                                                        random_state=42,
                                                        test_size=testSize)
    load_time = time.time() - start
    print('Bosch dataset loaded in %.2fs' % load_time, file=sys.stderr)
    return Data(X_train, X_test, y_train, y_test)

def prepare(dbFolder):
    return prepareImpl(dbFolder, 183747, True)


def metrics(y_test, y_prob):
    return classification_metrics_binary_prob(y_test, y_prob)

def catMetrics(y_test, y_prob):
    pred = np.argmax(y_prob, axis=1)
    return classification_metrics_binary_prob(y_test, pred)


nthreads = get_number_processors()
nTrees = 150

xgb_common_params = {
    "gamma":            0.1,
    "learning_rate":    0.1,
    "max_depth":        6,
    "max_leaves":       2**6,
    "min_child_weight": 1,
    "num_round":        nTrees,
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
}

lgb_common_params = {
    "learning_rate":    0.1,
    "min_child_weight": 1,
    "min_split_gain":   0.1,
    "num_leaves":       2**6,
    "num_round":        nTrees,
    "objective":        "binary",
    "reg_lambda":       1,
    "scale_pos_weight": 2,
    "subsample":        1,
    "task":             "train",
}

cat_common_params = {
    "depth":            6,
    "iterations":       nTrees,
    "l2_leaf_reg":      0.1,
    "learning_rate":    0.1,
    "loss_function":    "Logloss",
}

# NOTES: some benchmarks are disabled!
#  . cat-gpu throws the following error:
# _catboost.CatboostError: catboost/libs/algo/full_features.cpp:29: There are nans in test dataset (feature number 0) but there were not nans in learn dataset
#  . xgb-gpu  encounters illegal memory access
# [16:16:33] /xgboost/dmlc-core/include/dmlc/./logging.h:300: [16:16:33] /xgboost/src/tree/updater_gpu.cu:528: GPU plugin exception: /xgboost/src/tree/../common/device_helpers.cuh(319): an illegal memory access was encountered
# NOTES: some benchmarks are disabled!
benchmarks = {
    "xgb-cpu":      (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="exact",
                          nthread=nthreads)),
    "xgb-cpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, nthread=nthreads,
                          grow_policy="lossguide", tree_method="hist")),
    "xgb-gpu":      (False, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_exact",
                          objective="binary:logistic")),
    "xgb-gpu-hist": (True, XgbBenchmark, metrics,
                     dict(xgb_common_params, tree_method="gpu_hist",
                          objective="binary:logistic")),

    "lgbm-cpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, nthread=nthreads)),
    "lgbm-gpu":     (True, LgbBenchmark, metrics,
                     dict(lgb_common_params, device="gpu")),

    "cat-cpu":      (True, CatBenchmark, catMetrics,
                     dict(cat_common_params, thread_count=nthreads)),
    "cat-gpu":      (False, CatBenchmark, catMetrics,
                     dict(cat_common_params, task_type="GPU")),
}
