[![Work in Repl.it](https://classroom.github.com/assets/work-in-replit-14baed9a392b3a25080506f3b7b6d57f295ec2978f6f33ec97e36a161684cbe9.svg)](https://classroom.github.com/online_ide?assignment_repo_id=3683040&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```
## Plots

![plots](project/MNIST.png)

## Log

Find the same log in project/epoch_output.txt

(venv) asheinin@lfin0039 minitorch-4-arvens1990 % python project/run_mnist_multiclass.py
Setting up a new session...
Epoch  0  example  0  loss  37.09871267638074  accuracy  0.1
Epoch  0  example  800  loss  1838.4067973781448  accuracy  0.1
Epoch  0  example  1600  loss  1826.9856645361003  accuracy  0.275
Epoch  0  example  2400  loss  1675.4955136082865  accuracy  0.575
Epoch  0  example  3200  loss  1302.671893854108  accuracy  0.625
Epoch  0  example  4000  loss  1196.8006661543654  accuracy  0.7125
Epoch  0  example  4800  loss  1242.432258872231  accuracy  0.5
Epoch  1  example  0  loss  27.185664052129873  accuracy  0.4125
Epoch  1  example  800  loss  1302.5005552084267  accuracy  0.6
Epoch  1  example  1600  loss  1152.1595106995078  accuracy  0.75
Epoch  1  example  2400  loss  721.5266073074681  accuracy  0.7375
Epoch  1  example  3200  loss  567.5870628070893  accuracy  0.8125
Epoch  1  example  4000  loss  483.45537509901357  accuracy  0.8625
Epoch  1  example  4800  loss  476.64903382714954  accuracy  0.8625
Epoch  2  example  0  loss  5.334965488166319  accuracy  0.825
Epoch  2  example  800  loss  407.4777434921707  accuracy  0.875
Epoch  2  example  1600  loss  453.26279926304267  accuracy  0.85
Epoch  2  example  2400  loss  345.78130149654226  accuracy  0.8125
Epoch  2  example  3200  loss  334.9944349131333  accuracy  0.9
Epoch  2  example  4000  loss  334.3548282713806  accuracy  0.9
Epoch  2  example  4800  loss  323.0323328576326  accuracy  0.8375
Epoch  3  example  0  loss  1.5387684895424383  accuracy  0.8875
Epoch  3  example  800  loss  306.9062681034633  accuracy  0.8625
Epoch  3  example  1600  loss  360.1396793370539  accuracy  0.8375
Epoch  3  example  2400  loss  280.5899545740529  accuracy  0.8375
Epoch  3  example  3200  loss  301.5586490718894  accuracy  0.8875
Epoch  3  example  4000  loss  289.9282286838612  accuracy  0.9
Epoch  3  example  4800  loss  285.996507426817  accuracy  0.8625
Epoch  4  example  0  loss  1.6776933600483113  accuracy  0.8375
Epoch  4  example  800  loss  261.9758727928228  accuracy  0.875
Epoch  4  example  1600  loss  333.0043622417743  accuracy  0.85
Epoch  4  example  2400  loss  237.73887111920138  accuracy  0.8375
Epoch  4  example  3200  loss  292.91411889516615  accuracy  0.8875
Epoch  4  example  4000  loss  269.09794521095904  accuracy  0.8875
Epoch  4  example  4800  loss  269.5093175648871  accuracy  0.85
Epoch  5  example  0  loss  1.0934449395827919  accuracy  0.8625
Epoch  5  example  800  loss  224.19624251929574  accuracy  0.875
Epoch  5  example  1600  loss  333.36585633889166  accuracy  0.8625
Epoch  5  example  2400  loss  210.41019323488482  accuracy  0.825
Epoch  5  example  3200  loss  275.56918388758527  accuracy  0.8875
Epoch  5  example  4000  loss  255.20021357933803  accuracy  0.9
Epoch  5  example  4800  loss  240.77708362989443  accuracy  0.8375
Epoch  6  example  0  loss  1.8518126240840895  accuracy  0.8625
Epoch  6  example  800  loss  239.91068248126857  accuracy  0.8875
Epoch  6  example  1600  loss  326.3068187945117  accuracy  0.85
Epoch  6  example  2400  loss  181.78755070160136  accuracy  0.8625
Epoch  6  example  3200  loss  224.9277969161776  accuracy  0.8875
Epoch  6  example  4000  loss  229.44508241790848  accuracy  0.8625
Epoch  6  example  4800  loss  211.22245794010533  accuracy  0.85
Epoch  7  example  0  loss  0.757496273180513  accuracy  0.85
Epoch  7  example  800  loss  229.67370064243343  accuracy  0.9125
Epoch  7  example  1600  loss  292.78988336395014  accuracy  0.85
Epoch  7  example  2400  loss  149.9945158125942  accuracy  0.9
Epoch  7  example  3200  loss  206.95826527386302  accuracy  0.9125
Epoch  7  example  4000  loss  190.3588870670888  accuracy  0.9375
Epoch  7  example  4800  loss  199.4751768408688  accuracy  0.875
Epoch  8  example  0  loss  0.7822803062310868  accuracy  0.9125
Epoch  8  example  800  loss  197.6102224126013  accuracy  0.925
Epoch  8  example  1600  loss  269.88620244042323  accuracy  0.8625
Epoch  8  example  2400  loss  160.50627327753116  accuracy  0.9
Epoch  8  example  3200  loss  198.22503214381084  accuracy  0.9125
Epoch  8  example  4000  loss  201.93327398777078  accuracy  0.9125
Epoch  8  example  4800  loss  198.50565351332352  accuracy  0.8875
Epoch  9  example  0  loss  2.4019655102921558  accuracy  0.9
Epoch  9  example  800  loss  184.23752116254883  accuracy  0.8875
Epoch  9  example  1600  loss  259.9568959729168  accuracy  0.875
Epoch  9  example  2400  loss  199.08406196588507  accuracy  0.875
Epoch  9  example  3200  loss  218.4284808097267  accuracy  0.925
Epoch  9  example  4000  loss  201.30921222211006  accuracy  0.9
Epoch  9  example  4800  loss  184.03235419144715  accuracy  0.8875
Epoch  10  example  0  loss  1.3800832761356503  accuracy  0.8875
Epoch  10  example  800  loss  187.88141442710153  accuracy  0.8875
Epoch  10  example  1600  loss  241.78164057570962  accuracy  0.8625
Epoch  10  example  2400  loss  130.99019210500109  accuracy  0.8875
Epoch  10  example  3200  loss  156.96271346990443  accuracy  0.9125
Epoch  10  example  4000  loss  199.0205386299711  accuracy  0.875
Epoch  10  example  4800  loss  176.07700213579355  accuracy  0.875
Epoch  11  example  0  loss  0.9479734233025701  accuracy  0.8625
Epoch  11  example  800  loss  178.76171912342573  accuracy  0.9
Epoch  11  example  1600  loss  227.21867619842428  accuracy  0.8625
Epoch  11  example  2400  loss  118.26543916495812  accuracy  0.9125
Epoch  11  example  3200  loss  164.35351393234367  accuracy  0.925
Epoch  11  example  4000  loss  170.30315824947252  accuracy  0.875
Epoch  11  example  4800  loss  229.20776314301887  accuracy  0.85
Epoch  12  example  0  loss  0.3944895902462102  accuracy  0.8875
Epoch  12  example  800  loss  162.04589136763073  accuracy  0.9
Epoch  12  example  1600  loss  277.6930921411867  accuracy  0.825
Epoch  12  example  2400  loss  123.41753346555284  accuracy  0.8875
Epoch  12  example  3200  loss  135.12997120766903  accuracy  0.9125
Epoch  12  example  4000  loss  149.7669497093294  accuracy  0.9
Epoch  12  example  4800  loss  129.18577001849272  accuracy  0.8875
Epoch  13  example  0  loss  0.5456845502725125  accuracy  0.9125
Epoch  13  example  800  loss  124.48649133368254  accuracy  0.9125
Epoch  13  example  1600  loss  212.28989162037095  accuracy  0.9
Epoch  13  example  2400  loss  93.66996942250822  accuracy  0.9125
Epoch  13  example  3200  loss  124.96266319411927  accuracy  0.9375
Epoch  13  example  4000  loss  122.90178089469272  accuracy  0.925
Epoch  13  example  4800  loss  138.83645872343695  accuracy  0.8875
Epoch  14  example  0  loss  0.2266878308926943  accuracy  0.9125
Epoch  14  example  800  loss  144.4778646564424  accuracy  0.9
Epoch  14  example  1600  loss  186.05145464710395  accuracy  0.8875
Epoch  14  example  2400  loss  81.73952541828616  accuracy  0.9125
Epoch  14  example  3200  loss  126.95744016229747  accuracy  0.9375
Epoch  14  example  4000  loss  126.52888502662806  accuracy  0.9125
Epoch  14  example  4800  loss  128.10359034654374  accuracy  0.9125
Epoch  15  example  0  loss  0.3880455309269255  accuracy  0.9
Epoch  15  example  800  loss  131.86763360042522  accuracy  0.9125
Epoch  15  example  1600  loss  196.78276860895542  accuracy  0.875
Epoch  15  example  2400  loss  104.99500161760706  accuracy  0.925
Epoch  15  example  3200  loss  133.3720450651605  accuracy  0.925
Epoch  15  example  4000  loss  114.2102577466402  accuracy  0.925
Epoch  15  example  4800  loss  117.13184522634891  accuracy  0.9
Epoch  16  example  0  loss  0.6857898108983824  accuracy  0.9125
Epoch  16  example  800  loss  116.61258150237765  accuracy  0.9125
Epoch  16  example  1600  loss  204.25390530070257  accuracy  0.9
Epoch  16  example  2400  loss  76.32786036344672  accuracy  0.9375
Epoch  16  example  3200  loss  127.59108641393108  accuracy  0.925
Epoch  16  example  4000  loss  115.45654514905911  accuracy  0.925
Epoch  16  example  4800  loss  108.30762159274434  accuracy  0.9125
Epoch  17  example  0  loss  0.41238438862860693  accuracy  0.9
Epoch  17  example  800  loss  120.24953863571149  accuracy  0.9125
Epoch  17  example  1600  loss  160.94718442178873  accuracy  0.9125
Epoch  17  example  2400  loss  89.37867041918871  accuracy  0.925
Epoch  17  example  3200  loss  106.8677215674436  accuracy  0.9375
Epoch  17  example  4000  loss  99.29040936592544  accuracy  0.9375
Epoch  17  example  4800  loss  104.35234287271685  accuracy  0.9
Epoch  18  example  0  loss  0.8645477369825691  accuracy  0.925
Epoch  18  example  800  loss  102.60412111168887  accuracy  0.925
Epoch  18  example  1600  loss  165.05343417993288  accuracy  0.9
Epoch  18  example  2400  loss  81.4459402709417  accuracy  0.925
Epoch  18  example  3200  loss  109.28617878271518  accuracy  0.95
Epoch  18  example  4000  loss  95.71926180024411  accuracy  0.9375
Epoch  18  example  4800  loss  93.67986004194634  accuracy  0.9
Epoch  19  example  0  loss  0.31491277545123264  accuracy  0.8875
Epoch  19  example  800  loss  108.28162975142028  accuracy  0.9125
Epoch  19  example  1600  loss  147.16935570886497  accuracy  0.9
Epoch  19  example  2400  loss  78.95901851899656  accuracy  0.9375
Epoch  19  example  3200  loss  103.10020770857743  accuracy  0.95
Epoch  19  example  4000  loss  97.89310031291758  accuracy  0.95
Epoch  19  example  4800  loss  123.96337132202896  accuracy  0.925
Epoch  20  example  0  loss  1.3893005874935662  accuracy  0.925
Epoch  20  example  800  loss  96.8567309444049  accuracy  0.9125
Epoch  20  example  1600  loss  132.13697653177425  accuracy  0.875
Epoch  20  example  2400  loss  77.08032242957702  accuracy  0.925
Epoch  20  example  3200  loss  116.12700599150071  accuracy  0.9625
Epoch  20  example  4000  loss  107.21852438800953  accuracy  0.925
Epoch  20  example  4800  loss  117.69328468315193  accuracy  0.925
Epoch  21  example  0  loss  0.5962517820407278  accuracy  0.925
Epoch  21  example  800  loss  98.17688596762235  accuracy  0.9125
Epoch  21  example  1600  loss  135.88514411073703  accuracy  0.9
Epoch  21  example  2400  loss  75.90070843654979  accuracy  0.925
Epoch  21  example  3200  loss  84.88835921278637  accuracy  0.9375
Epoch  21  example  4000  loss  89.23016481943435  accuracy  0.95
Epoch  21  example  4800  loss  97.46406068076234  accuracy  0.9375
Epoch  22  example  0  loss  0.5114182548820252  accuracy  0.9
Epoch  22  example  800  loss  102.73631071970675  accuracy  0.9125
Epoch  22  example  1600  loss  132.19984008801663  accuracy  0.9
Epoch  22  example  2400  loss  82.94666042022152  accuracy  0.9125
Epoch  22  example  3200  loss  82.46163330702959  accuracy  0.95
Epoch  22  example  4000  loss  76.20302883816497  accuracy  0.925
Epoch  22  example  4800  loss  85.0631048863999  accuracy  0.925
Epoch  23  example  0  loss  0.5677882972776231  accuracy  0.925
Epoch  23  example  800  loss  100.35845868535728  accuracy  0.925
Epoch  23  example  1600  loss  132.20532858418738  accuracy  0.9125
Epoch  23  example  2400  loss  61.1306442964299  accuracy  0.95
Epoch  23  example  3200  loss  86.41632522721953  accuracy  0.95
Epoch  23  example  4000  loss  84.56597205217805  accuracy  0.9375
Epoch  23  example  4800  loss  101.86885554564553  accuracy  0.925
Epoch  24  example  0  loss  0.5014314673584748  accuracy  0.9125
Epoch  24  example  800  loss  87.2152577147676  accuracy  0.925
Epoch  24  example  1600  loss  116.98204293576936  accuracy  0.875
Epoch  24  example  2400  loss  77.54794819111788  accuracy  0.9125
Epoch  24  example  3200  loss  93.26787158009026  accuracy  0.9375
Epoch  24  example  4000  loss  84.06547222336808  accuracy  0.925
Epoch  24  example  4800  loss  107.41401823161962  accuracy  0.9
Epoch  25  example  0  loss  0.6917855539464726  accuracy  0.9125
Epoch  25  example  800  loss  93.37867816175817  accuracy  0.925
Epoch  25  example  1600  loss  118.0235463751415  accuracy  0.8875
Epoch  25  example  2400  loss  70.57567371221079  accuracy  0.9125
Epoch  25  example  3200  loss  91.64336310037704  accuracy  0.9125
Epoch  25  example  4000  loss  74.3170660842899  accuracy  0.925
Epoch  25  example  4800  loss  101.97907926947963  accuracy  0.925
Epoch  26  example  0  loss  0.5836149956614793  accuracy  0.9375
Epoch  26  example  800  loss  98.81179799586876  accuracy  0.9
Epoch  26  example  1600  loss  114.34956804524973  accuracy  0.9
Epoch  26  example  2400  loss  73.29748096334487  accuracy  0.9125
Epoch  26  example  3200  loss  91.21750199074063  accuracy  0.925
Epoch  26  example  4000  loss  71.69258311274916  accuracy  0.925
Epoch  26  example  4800  loss  93.730819504679  accuracy  0.9125
Epoch  27  example  0  loss  0.7312173652841807  accuracy  0.9
Epoch  27  example  800  loss  103.17065689816917  accuracy  0.9125
Epoch  27  example  1600  loss  101.1935508106112  accuracy  0.8875
Epoch  27  example  2400  loss  72.53046822861313  accuracy  0.9
Epoch  27  example  3200  loss  85.28600780237223  accuracy  0.9375
Epoch  27  example  4000  loss  79.25593302744576  accuracy  0.9125
Epoch  27  example  4800  loss  79.2370559567418  accuracy  0.9
Epoch  28  example  0  loss  0.20842914044805116  accuracy  0.9
Epoch  28  example  800  loss  88.69680234601157  accuracy  0.9
Epoch  28  example  1600  loss  114.19898846966247  accuracy  0.8875
Epoch  28  example  2400  loss  64.53632732890388  accuracy  0.9125
Epoch  28  example  3200  loss  73.9097401590563  accuracy  0.9125
Epoch  28  example  4000  loss  68.78187222184623  accuracy  0.9375
Epoch  28  example  4800  loss  72.4963704145172  accuracy  0.925
Epoch  29  example  0  loss  0.12951368531727603  accuracy  0.9125
Epoch  29  example  800  loss  90.16143815550964  accuracy  0.9125
Epoch  29  example  1600  loss  113.09589594263889  accuracy  0.8875
Epoch  29  example  2400  loss  67.20135207608416  accuracy  0.9125
Epoch  29  example  3200  loss  74.43792738843845  accuracy  0.925
Epoch  29  example  4000  loss  63.47021058393665  accuracy  0.925
Epoch  29  example  4800  loss  72.28059066221644  accuracy  0.925
Epoch  30  example  0  loss  0.28427833990797424  accuracy  0.925
Epoch  30  example  800  loss  78.45244824344665  accuracy  0.9
Epoch  30  example  1600  loss  124.38755455515673  accuracy  0.8875
Epoch  30  example  2400  loss  55.59093318517467  accuracy  0.925
Epoch  30  example  3200  loss  61.03329590680961  accuracy  0.925
Epoch  30  example  4000  loss  56.96321756332909  accuracy  0.95
Epoch  30  example  4800  loss  74.35083912906971  accuracy  0.9125
Epoch  31  example  0  loss  0.7226482746861325  accuracy  0.9125
Epoch  31  example  800  loss  69.54859625386442  accuracy  0.9125
Epoch  31  example  1600  loss  104.42818865246727  accuracy  0.8875
Epoch  31  example  2400  loss  58.07605474386533  accuracy  0.9125
Epoch  31  example  3200  loss  78.23641374077003  accuracy  0.9125
Epoch  31  example  4000  loss  58.325074073596795  accuracy  0.9125
Epoch  31  example  4800  loss  60.48907436845285  accuracy  0.9125
Epoch  32  example  0  loss  0.7751496937280531  accuracy  0.8875
Epoch  32  example  800  loss  81.25671885959086  accuracy  0.9125
Epoch  32  example  1600  loss  93.11745370734596  accuracy  0.875
Epoch  32  example  2400  loss  72.82609915088453  accuracy  0.925
Epoch  32  example  3200  loss  70.51974193894819  accuracy  0.9125
Epoch  32  example  4000  loss  67.76003625341016  accuracy  0.9125
Epoch  32  example  4800  loss  94.38337387475826  accuracy  0.925
Epoch  33  example  0  loss  0.1262333203350785  accuracy  0.9125
Epoch  33  example  800  loss  79.66643838655152  accuracy  0.925
Epoch  33  example  1600  loss  91.91525609840629  accuracy  0.8875
Epoch  33  example  2400  loss  49.34179636205149  accuracy  0.925
Epoch  33  example  3200  loss  54.073214702697534  accuracy  0.925
Epoch  33  example  4000  loss  53.070158149795674  accuracy  0.9375
Epoch  33  example  4800  loss  71.00744333552132  accuracy  0.9125
Epoch  34  example  0  loss  0.11080759434669485  accuracy  0.9125
Epoch  34  example  800  loss  74.51355032929797  accuracy  0.9
Epoch  34  example  1600  loss  98.70506094223097  accuracy  0.9125
Epoch  34  example  2400  loss  63.500642385467515  accuracy  0.9
Epoch  34  example  3200  loss  64.16766843451563  accuracy  0.9375
Epoch  34  example  4000  loss  44.906599049706365  accuracy  0.9125
Epoch  34  example  4800  loss  71.11645212136268  accuracy  0.9125
Epoch  35  example  0  loss  0.06726874676012162  accuracy  0.9
Epoch  35  example  800  loss  72.24080922952673  accuracy  0.9125
Epoch  35  example  1600  loss  87.2774141088471  accuracy  0.875
Epoch  35  example  2400  loss  59.40857100425721  accuracy  0.9125
Epoch  35  example  3200  loss  62.178357198046236  accuracy  0.9
Epoch  35  example  4000  loss  50.69233252360511  accuracy  0.925
Epoch  35  example  4800  loss  70.36168127241577  accuracy  0.9
Epoch  36  example  0  loss  0.11966742900008676  accuracy  0.9
Epoch  36  example  800  loss  66.75390006173893  accuracy  0.925
Epoch  36  example  1600  loss  68.71710184574289  accuracy  0.8875
Epoch  36  example  2400  loss  43.06223982965604  accuracy  0.9125
Epoch  36  example  3200  loss  65.88228679997809  accuracy  0.925
Epoch  36  example  4000  loss  53.41704983646596  accuracy  0.925
Epoch  36  example  4800  loss  46.67881370270472  accuracy  0.925
Epoch  37  example  0  loss  0.06490073246588679  accuracy  0.925
Epoch  37  example  800  loss  60.40768142324805  accuracy  0.925
Epoch  37  example  1600  loss  84.68048174131951  accuracy  0.875
Epoch  37  example  2400  loss  33.2876417576896  accuracy  0.925
Epoch  37  example  3200  loss  48.392274593476245  accuracy  0.925
Epoch  37  example  4000  loss  61.53328482822786  accuracy  0.9125
Epoch  37  example  4800  loss  56.43853928349495  accuracy  0.9375
Epoch  38  example  0  loss  0.06622421449300653  accuracy  0.925
Epoch  38  example  800  loss  70.89053352118134  accuracy  0.925
Epoch  38  example  1600  loss  73.26250246473806  accuracy  0.9125
Epoch  38  example  2400  loss  52.298762242823415  accuracy  0.9125
Epoch  38  example  3200  loss  76.88192103722739  accuracy  0.9125
Epoch  38  example  4000  loss  61.382243788732055  accuracy  0.9375
Epoch  38  example  4800  loss  75.54352577031658  accuracy  0.9375
Epoch  39  example  0  loss  0.2741790974071483  accuracy  0.9375
Epoch  39  example  800  loss  76.71723006378863  accuracy  0.9125
Epoch  39  example  1600  loss  73.69988535092845  accuracy  0.9125
Epoch  39  example  2400  loss  41.58106412006707  accuracy  0.9125
Epoch  39  example  3200  loss  51.56366296414041  accuracy  0.9125
Epoch  39  example  4000  loss  45.199023272804176  accuracy  0.925
Epoch  39  example  4800  loss  59.28950206873887  accuracy  0.9125
Epoch  40  example  0  loss  0.0513680542842625  accuracy  0.8875
Epoch  40  example  800  loss  54.432810308056  accuracy  0.9125
Epoch  40  example  1600  loss  56.73044801790054  accuracy  0.9125
Epoch  40  example  2400  loss  45.50395154487208  accuracy  0.925
Epoch  40  example  3200  loss  46.29372519121839  accuracy  0.9125
Epoch  40  example  4000  loss  44.271962985204254  accuracy  0.95
Epoch  40  example  4800  loss  57.672233609009766  accuracy  0.9125
Epoch  41  example  0  loss  0.1586270038632076  accuracy  0.9
Epoch  41  example  800  loss  51.245311657918414  accuracy  0.925
Epoch  41  example  1600  loss  56.99267557475619  accuracy  0.8875
Epoch  41  example  2400  loss  38.3644193124456  accuracy  0.9125
Epoch  41  example  3200  loss  50.918131554224445  accuracy  0.925
Epoch  41  example  4000  loss  50.05361024278014  accuracy  0.925
Epoch  41  example  4800  loss  47.31531892037775  accuracy  0.9375
Epoch  42  example  0  loss  0.008069533136511708  accuracy  0.925
Epoch  42  example  800  loss  48.52132753111596  accuracy  0.925
Epoch  42  example  1600  loss  67.12315298196191  accuracy  0.9
Epoch  42  example  2400  loss  33.12958540758634  accuracy  0.9125
Epoch  42  example  3200  loss  35.59913689156429  accuracy  0.925
Epoch  42  example  4000  loss  50.74114989852647  accuracy  0.925
Epoch  42  example  4800  loss  41.070870223936595  accuracy  0.9
Epoch  43  example  0  loss  0.05401228589364049  accuracy  0.925
Epoch  43  example  800  loss  36.690897145661786  accuracy  0.9375
Epoch  43  example  1600  loss  47.97371649513883  accuracy  0.9
Epoch  43  example  2400  loss  37.25458377815141  accuracy  0.9125
Epoch  43  example  3200  loss  49.544471259769544  accuracy  0.925
Epoch  43  example  4000  loss  45.24895303055171  accuracy  0.925
Epoch  43  example  4800  loss  57.09469192617212  accuracy  0.9375
Epoch  44  example  0  loss  0.3346205378178766  accuracy  0.9375
Epoch  44  example  800  loss  68.79377879550687  accuracy  0.925
Epoch  44  example  1600  loss  77.80582098814986  accuracy  0.9125
Epoch  44  example  2400  loss  93.58368310704982  accuracy  0.95
Epoch  44  example  3200  loss  114.78561084849434  accuracy  0.925
Epoch  44  example  4000  loss  75.0587179777045  accuracy  0.925
