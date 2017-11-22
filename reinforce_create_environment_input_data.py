from abb_deeplearning.abb_data_pipeline.abb_clouddrl_transformation_pipeline import create_rl_environment_input_files
from abb_deeplearning.abb_data_pipeline import abb_clouddrl_constants as ac



create_rl_environment_input_files(automatic_daytime=True,file_filter={"Resize_sp_256", ".jpeg"},output_path="",output_name="rl_data_sp.csv")