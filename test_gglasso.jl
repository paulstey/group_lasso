############
# Testing Group-Wise Lasso Using GMD Algorithm
# Translated from Yang & Zou's Fortran code
# Date: Dec 17, 2015
# Author: Paul Stey
############

using DataFrames
using Debug



include("C:/Users/Pstey/juliawd/group_lasso/gglasso.jl")


# d_raw = readcsv("/home/ubuntu/rwd/bardet.csv", header = true)
d_raw = readcsv("C:/Users/Pstey/juliawd/group_lasso/bardet.csv", header = true)

d = d_raw[1]

X = d[:, 1:100]
y = d[:, 101]




group1 = rep(1:20, each = 5)



@time z = grp_lasso(X, y, group1);



