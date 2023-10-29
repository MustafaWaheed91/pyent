from unittest import TestCase
import unittest


class EntityMatchTest(TestCase):
    """
        Crate unit tests for each of the components in the 
        solution, this includes datasets, features, and train modules

        :param TestCase (unittest.TestCase): Methods that provides testing suite and
    """
    x = None

    @classmethod
    def setUpClass(cls) -> None:
        """
            :return:
        """
        cls.x = 'ab'

    def test_create_dataframes(self):
        """
            this function will test the expected dummy data frame
            with candidate pairs based on attribute blocking
        """
        self.assertEquals(self.x, 'ab')

    @classmethod
    def tearDownClass(cls) -> None:
        """
            :return:
        """
        del cls.x


if __name__ == "__main__":
    unittest.main()
