#!/bin/bash
# This shell script downloads all available pre-trained weights for MS^2 dataset 
# if you want to acess individual weight through website, use below links.

# Pre-trained weights for Monocular Depth Networks
# DORN
# https://www.dropbox.com/scl/fi/ssh2pb95puanrj230i386/MS2_MD_DORN_RGB_ckpt.ckpt?rlkey=29gt7i563wlr50y2ess6x4kqe&st=c9c5mkoi&dl=0
# https://www.dropbox.com/scl/fi/wy94xjloxjo821gzyzhnu/MS2_MD_DORN_NIR_ckpt.ckpt?rlkey=i7wus22uvqka7ky70a6yvf3f3&st=4v00zct7&dl=0
# https://www.dropbox.com/scl/fi/b8m0iwk988l6lwcwi5m0o/MS2_MD_DORN_THR_ckpt.ckpt?rlkey=j0sfm0xmpmokdlkgb7cqzhb08&st=aeego6v0&dl=0
# BTS
# https://www.dropbox.com/scl/fi/dm19tpr22td05mibqvv0j/MS2_MD_BTS_RGB_ckpt.ckpt?rlkey=5u6ejjqboj8kri2o7x7ej1ari&st=blhwffbh&dl=0
# https://www.dropbox.com/scl/fi/etmoypbdmd3sf4wcsefhe/MS2_MD_BTS_NIR_ckpt.ckpt?rlkey=hmq9iw1ojrx2e0sgrnpj0gjja&st=bzh3xgyv&dl=0
# https://www.dropbox.com/scl/fi/gqko0ownuccirzy1mg1ug/MS2_MD_BTS_THR_ckpt.ckpt?rlkey=kn9v8mceycpydd6i8fgy2k0u8&st=bc1qbquf&dl=0
# AdaBins
# https://www.dropbox.com/scl/fi/exjo1vq8ygzfut733tvmi/MS2_MD_AdaBins_RGB_ckpt.ckpt?rlkey=1m0qq7tqyieurfsrpmdganw8q&st=jcxumgw6&dl=0
# https://www.dropbox.com/scl/fi/1r78opyyj1q8u4w5xq79m/MS2_MD_AdaBins_NIR_ckpt.ckpt?rlkey=ta9051eqi4x0o2b0ozftpywsp&st=904iricc&dl=0
# https://www.dropbox.com/scl/fi/2ar82el6mdm4myumzy175/MS2_MD_AdaBins_THR_ckpt.ckpt?rlkey=zt1i4vapcbq6xpeu1659grjbi&st=pizidekp&dl=0
# DPT large
# https://www.dropbox.com/scl/fi/h14so4nqwzmxwi416t02h/MS2_MD_DPT_Large_RGB_ckpt.ckpt?rlkey=2synx4yk41d6unw1ak3o5fhvg&st=q05wkxnm&dl=0
# https://www.dropbox.com/scl/fi/eq39o6bk6dnkq2rol1v3r/MS2_MD_DPT_Large_NIR_ckpt.ckpt?rlkey=pgii8jmm02d6swcrvnxac2dw2&st=ocm194ag&dl=0
# https://www.dropbox.com/scl/fi/w8fsacxycqy065hvulvmk/MS2_MD_DPT_Large_THR_ckpt.ckpt?rlkey=gl6n35oduuandg4m2ndkc3qjk&st=9mu5hbnq&dl=0
# NewCRF
# https://www.dropbox.com/scl/fi/b332awotceos6jxb25w38/MS2_MD_NeWCRF_RGB_ckpt.ckpt?rlkey=8hpgg8g1ex3m1208y73kj5ltp&st=firpnxua&dl=0
# https://www.dropbox.com/scl/fi/e48k5azsb08lp65lpcagl/MS2_MD_NeWCRF_NIR_ckpt.ckpt?rlkey=vg7kjhmfpw564sbyjsbgr51eu&st=3vd3dmbe&dl=0
# https://www.dropbox.com/scl/fi/mn9jm68p08kdwl89hls82/MS2_MD_NeWCRF_THR_ckpt.ckpt?rlkey=g3kzyc2f4y3t9q7blrmc2j2h0&st=t8p6mcu8&dl=0

# Pre-trained weights for Stereo Matching Networks
# PSMNet
# https://www.dropbox.com/scl/fi/ir342h44t1tvw36qmul8a/MS2_SM_PSMNet_RGB_ckpt.ckpt?rlkey=w578ciw1wzc72i1yriusiu4dg&st=3votfzs3&dl=0
# https://www.dropbox.com/scl/fi/vsrqwe9guzeenviwueu2u/MS2_SM_PSMNet_THR_ckpt.ckpt?rlkey=lha975k05b5bmg2bhp7hc3ez8&st=9a3g8cym&dl=0
# GWCNet
# https://www.dropbox.com/scl/fi/3xdvul8peuk4vmpthzxnu/MS2_SM_GWCNet_RGB_ckpt.ckpt?rlkey=43vpzr6xfyyixddbl4ms5af02&st=zwqlk8mz&dl=0
# https://www.dropbox.com/scl/fi/q94yfl3l0g76ygm7ieaug/MS2_SM_GWCNet_THR_ckpt.ckpt?rlkey=zpqm52b7zz1gpwc32y4wz4gc5&st=58vv7fxz&dl=0
# AANet
# https://www.dropbox.com/scl/fi/t8x84ke3glfnk9m07xqak/MS2_SM_AANet_RGB_ckpt.ckpt?rlkey=zo2dphp58f4egbzcsvwjt94rm&st=imy2xf3c&dl=0
# https://www.dropbox.com/scl/fi/tdxyzvpw9wlc3dbp0a1dn/MS2_SM_AANet_THR_ckpt.ckpt?rlkey=j365ti4fcyh09peifme5vbedd&st=n3iywazw&dl=0
# ACVNet
# https://www.dropbox.com/scl/fi/99ql7wj6k3j8rploittyc/MS2_SM_ACVNet_RGB_ckpt.ckpt?rlkey=lrkjrvutaf79d8bwyrtpub6zm&st=ujtxvseu&dl=0
# https://www.dropbox.com/scl/fi/rvrhwe486u03ls241foe5/MS2_SM_ACVNet_THR_ckpt.ckpt?rlkey=jniq0nbrblin72dqpjszc3wxz&st=uqhia4q5&dl=0

wget --tries=2 -c -O MS2_MD_DORN_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/ssh2pb95puanrj230i386/MS2_MD_DORN_RGB_ckpt.ckpt?rlkey=29gt7i563wlr50y2ess6x4kqe&st=c9c5mkoi"
wget --tries=2 -c -O MS2_MD_DORN_NIR_ckpt.ckpt "https://www.dropbox.com/scl/fi/wy94xjloxjo821gzyzhnu/MS2_MD_DORN_NIR_ckpt.ckpt?rlkey=i7wus22uvqka7ky70a6yvf3f3&st=4v00zct7"
wget --tries=2 -c -O MS2_MD_DORN_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/b8m0iwk988l6lwcwi5m0o/MS2_MD_DORN_THR_ckpt.ckpt?rlkey=j0sfm0xmpmokdlkgb7cqzhb08&st=aeego6v0"
wget --tries=2 -c -O MS2_MD_BTS_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/dm19tpr22td05mibqvv0j/MS2_MD_BTS_RGB_ckpt.ckpt?rlkey=5u6ejjqboj8kri2o7x7ej1ari&st=blhwffbh"
wget --tries=2 -c -O MS2_MD_BTS_NIR_ckpt.ckpt "https://www.dropbox.com/scl/fi/etmoypbdmd3sf4wcsefhe/MS2_MD_BTS_NIR_ckpt.ckpt?rlkey=hmq9iw1ojrx2e0sgrnpj0gjja&st=bzh3xgyv"
wget --tries=2 -c -O MS2_MD_BTS_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/gqko0ownuccirzy1mg1ug/MS2_MD_BTS_THR_ckpt.ckpt?rlkey=kn9v8mceycpydd6i8fgy2k0u8&st=bc1qbquf"
wget --tries=2 -c -O MS2_MD_AdaBins_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/exjo1vq8ygzfut733tvmi/MS2_MD_AdaBins_RGB_ckpt.ckpt?rlkey=1m0qq7tqyieurfsrpmdganw8q&st=jcxumgw6"
wget --tries=2 -c -O MS2_MD_AdaBins_NIR_ckpt.ckpt "https://www.dropbox.com/scl/fi/1r78opyyj1q8u4w5xq79m/MS2_MD_AdaBins_NIR_ckpt.ckpt?rlkey=ta9051eqi4x0o2b0ozftpywsp&st=904iricc"
wget --tries=2 -c -O MS2_MD_AdaBins_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/2ar82el6mdm4myumzy175/MS2_MD_AdaBins_THR_ckpt.ckpt?rlkey=zt1i4vapcbq6xpeu1659grjbi&st=pizidekp"
wget --tries=2 -c -O MS2_MD_DPT_Large_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/h14so4nqwzmxwi416t02h/MS2_MD_DPT_Large_RGB_ckpt.ckpt?rlkey=2synx4yk41d6unw1ak3o5fhvg&st=q05wkxnm"
wget --tries=2 -c -O MS2_MD_DPT_Large_NIR_ckpt.ckpt "https://www.dropbox.com/scl/fi/eq39o6bk6dnkq2rol1v3r/MS2_MD_DPT_Large_NIR_ckpt.ckpt?rlkey=pgii8jmm02d6swcrvnxac2dw2&st=ocm194ag"
wget --tries=2 -c -O MS2_MD_DPT_Large_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/w8fsacxycqy065hvulvmk/MS2_MD_DPT_Large_THR_ckpt.ckpt?rlkey=gl6n35oduuandg4m2ndkc3qjk&st=9mu5hbnq"
wget --tries=2 -c -O MS2_MD_NeWCRF_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/b332awotceos6jxb25w38/MS2_MD_NeWCRF_RGB_ckpt.ckpt?rlkey=8hpgg8g1ex3m1208y73kj5ltp&st=firpnxua"
wget --tries=2 -c -O MS2_MD_NeWCRF_NIR_ckpt.ckpt "https://www.dropbox.com/scl/fi/e48k5azsb08lp65lpcagl/MS2_MD_NeWCRF_NIR_ckpt.ckpt?rlkey=vg7kjhmfpw564sbyjsbgr51eu&st=3vd3dmbe"
wget --tries=2 -c -O MS2_MD_NeWCRF_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/mn9jm68p08kdwl89hls82/MS2_MD_NeWCRF_THR_ckpt.ckpt?rlkey=g3kzyc2f4y3t9q7blrmc2j2h0&st=t8p6mcu8"

wget --tries=2 -c -O MS2_SM_PSMNet_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/ir342h44t1tvw36qmul8a/MS2_SM_PSMNet_RGB_ckpt.ckpt?rlkey=w578ciw1wzc72i1yriusiu4dg&st=3votfzs3"
wget --tries=2 -c -O MS2_SM_PSMNet_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/vsrqwe9guzeenviwueu2u/MS2_SM_PSMNet_THR_ckpt.ckpt?rlkey=lha975k05b5bmg2bhp7hc3ez8&st=9a3g8cym"
wget --tries=2 -c -O MS2_SM_GWCNet_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/3xdvul8peuk4vmpthzxnu/MS2_SM_GWCNet_RGB_ckpt.ckpt?rlkey=43vpzr6xfyyixddbl4ms5af02&st=zwqlk8mz"
wget --tries=2 -c -O MS2_SM_GWCNet_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/q94yfl3l0g76ygm7ieaug/MS2_SM_GWCNet_THR_ckpt.ckpt?rlkey=zpqm52b7zz1gpwc32y4wz4gc5&st=58vv7fxz"
wget --tries=2 -c -O MS2_SM_AANet_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/t8x84ke3glfnk9m07xqak/MS2_SM_AANet_RGB_ckpt.ckpt?rlkey=zo2dphp58f4egbzcsvwjt94rm&st=imy2xf3c"
wget --tries=2 -c -O MS2_SM_AANet_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/tdxyzvpw9wlc3dbp0a1dn/MS2_SM_AANet_THR_ckpt.ckpt?rlkey=j365ti4fcyh09peifme5vbedd&st=n3iywazw"
wget --tries=2 -c -O MS2_SM_ACVNet_RGB_ckpt.ckpt "https://www.dropbox.com/scl/fi/99ql7wj6k3j8rploittyc/MS2_SM_ACVNet_RGB_ckpt.ckpt?rlkey=lrkjrvutaf79d8bwyrtpub6zm&st=ujtxvseu"
wget --tries=2 -c -O MS2_SM_ACVNet_THR_ckpt.ckpt "https://www.dropbox.com/scl/fi/rvrhwe486u03ls241foe5/MS2_SM_ACVNet_THR_ckpt.ckpt?rlkey=jniq0nbrblin72dqpjszc3wxz&st=uqhia4q5"

