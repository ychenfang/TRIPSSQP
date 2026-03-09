TRIPSSQP = Parameter.TRIPSSQPParams(true,
                    100000,                    #Max_Iter
                    1e-4,                   #EPS_Res
                    1.0,                    #mu
                    1.0,                    #delta_k
                    1.0,                    #epsilon_k
                    1,                    #theta_k
                    1,                      #TotalRep
                    1.5,                    #rho
                    1.5,                    #gamma
                    10.0,                    #delta_max
                    0.01,                   #kap_g
                    0.01,                   #kap_h
                    0.0005,                   #kap_f
                    0.9,                    #epsilon_s
                    0.5,                   #xi
                    0.05,                    #p_h
                    0.05,                    #p_g
                    0.05,                    #p_f
                    0.6,                    #eta
                    [5],                      # C_grad
                    [1e-8,1e-4,1e-2,1e-1])  #Sigma
