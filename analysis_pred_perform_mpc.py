


from abb_deeplearning.abb_mpc_controller import abb_mpc
from abb_deeplearning.abb_data_pipeline.abb_clouddrl_read_pipeline import read_full_int_irr_data
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("pred_path", help="date to show")
parser.add_argument("change_constraint", help="date to show") #100
parser.add_argument("skip_pred", help="date to show") #1
parser.add_argument("nr_pred", help="date to show") #11
parser.add_argument("set_path",help="set which set")
args = parser.parse_args()



pred_path = args.pred_path
change_constraint = int(args.change_constraint)
nr_pred=int(args.nr_pred)


skip_preds = int(args.skip_pred) # how many predictions are skipped for next mpc step. if 1 then mps_step lasts for about 6-8 seconds
set_path = str(args.set_path)

# attention: should not be larger than the horizon you have in your prediction (10 mins ..  so roughly 600/6 = 100 roughly 6 sec per sample)
print("SKIP_PRED",skip_preds)
full_irr_data = read_full_int_irr_data()

with open(set_path) as f:
    days = f.readlines()
    day_list = [l.strip() for l in days]



abb_mpc.__prediction_mpc__(
    prediction_path=pred_path,day_list=day_list,nr_predictions=nr_pred,plot=False,full_int_irr_pd=full_irr_data,skip_preds=skip_preds,change_constraint_wh_min=change_constraint)
