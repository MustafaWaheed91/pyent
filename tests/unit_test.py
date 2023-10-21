from unittest import TestCase
import unittest


class EntityMatchTest(TestCase):
    """
        Crate unit tests for each of the components in the 
        solution, this includes datasets, features, and train modules

    :param TestCase (unittest.TestCase): 
        Methods that provides testing suite and 
    """
    
    @classmethod
    def setUp(self) -> None:
        self.x = 'ab'


    def test_create_dataframes(self):
        """
            this function will test the expected dummy data frame
            with candidate pairs based on attribute blocking
        """ 
        assert self.x == 'ab'


    def tearDown(self) -> None:
        del self.x


if __name__ == "__main__":
    unittest.main()
