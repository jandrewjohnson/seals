import os
import hazelbean as hb

import seals_generate_base_data

p = hb.ProjectFlow('../../projects/seals_generate_base_data_mosaic_is_natural')
p.esa_years_to_convert = [2015]
# p.esa_years_to_convert = [2000, 2010, 2014, 2015]
p.classify_mosaic_as_natural = 1


p.generated_kernels_task = p.add_task(seals_generate_base_data.generated_kernels)
p.lulc_simplifications_task = p.add_task(seals_generate_base_data.lulc_simplifications)
p.lulc_binaries_task = p.add_task(seals_generate_base_data.lulc_binaries)
p.lulc_convolutions_task = p.add_task(seals_generate_base_data.lulc_convolutions)

p.generated_kernels_task.run = 0
p.lulc_simplifications_task.run = 1
p.lulc_binaries_task.run = 0
p.lulc_convolutions_task.run = 0

if __name__ == '__main__':
    p.execute()

