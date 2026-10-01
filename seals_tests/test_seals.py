import unittest, os
import hazelbean as hb

import seals
from seals import run_seals_standard

class TestSEALS(unittest.TestCase):

    def test_seals_standard_fast(self):
        """Run the standard SEALS pipeline through run_seals_standard.run_project and check the final stitched LULC exists."""

        # Same entry point as run_seals_standard.py -- only the placement differs.
        # run_mode: 'fresh_intermediate' rebuilds all computation each run but keeps input/.
        p = hb.ProjectFlow(project_name='test_standard_fast', run_mode='fresh_intermediate',
                           extra_dirs=['Files', 'seals', 'projects', 'tests'])

        # ProjectFlow looks for input_template/ beside the calling script, which here is
        # seals_tests/. Point it at the package's template so the test reads the tracked CSVs
        # instead of falling through to the base_data copy.
        p.input_template_dir = os.path.join(os.path.dirname(seals.__file__), 'input_template')
        p.scenario_definitions_filename = 'standard_scenarios.csv'

        run_seals_standard.run_project(p)

        required_result_path = os.path.join(p.stitched_lulc_simplified_scenarios_dir, 'lulc_esa_seals7_ssp2_rcp45_luh2-message_bau_shift_2050.tif')
        self.assertTrue(hb.path_exists(required_result_path))


if __name__ == "__main__":
    unittest.main()
