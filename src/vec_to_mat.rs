pub fn get_col_from_index(index: usize, num_row: usize) -> usize {
    index / num_row
}
pub fn get_row_from_index(index: usize, num_row: usize) -> usize {
    index % num_row
}

/*
pub fn get_element_from_matrix<T>(
    row_num:usize,
    col_num:usize,
    num_rows:usize,
    array:&[T]
)->&T{
    &array[col_num*num_rows+row_num]
}*/

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn col_from_index_correctly_gets_index() {
        let result = get_col_from_index(3, 2);
        assert_eq!(result, 1); //zero based
    }
    #[test]
    fn col_from_index_correctly_gets_index_at_begin() {
        let result = get_col_from_index(2, 2);
        assert_eq!(result, 1); //zero based
    }
    #[test]
    fn col_from_index_correctly_gets_index_at_end() {
        let result = get_col_from_index(1, 2);
        assert_eq!(result, 0); //zero based
    }
    #[test]
    fn col_from_index_correctly_gets_index_one_row() {
        let result = get_col_from_index(1, 1);
        assert_eq!(result, 1); //zero based
    }
    #[test]
    fn col_from_index_correctly_gets_index_one_row_two() {
        let result = get_col_from_index(2, 1);
        assert_eq!(result, 2); //zero based
    }
    #[test]
    fn row_from_index_correctly_gets_index() {
        let result = get_row_from_index(3, 2);
        assert_eq!(result, 1); //zero based
    }
    #[test]
    fn row_from_index_correctly_gets_index_at_begin() {
        let result = get_row_from_index(2, 2);
        assert_eq!(result, 0); //zero based
    }
    #[test]
    fn row_from_index_correctly_gets_index_at_end() {
        let result = get_row_from_index(1, 2);
        assert_eq!(result, 1); //zero based
    }
    #[test]
    fn row_from_index_correctly_gets_index_one_row() {
        let result = get_row_from_index(1, 1);
        assert_eq!(result, 0); //zero based
    }
    #[test]
    fn row_from_index_correctly_gets_index_one_row_two() {
        let result = get_row_from_index(2, 1);
        assert_eq!(result, 0); //zero based
    }
    /*#[test]
    fn test_get_two_d_array(){
        let arr=vec![1, 2, 3, 4, 5, 6];
        let num_rows=2;
        let row_index_1=1;
        let col_index_1=0;
        let row_index_2=0;
        let col_index_2=1;
        assert_eq!(*get_element_from_matrix(
            row_index_1, col_index_1,
            num_rows, &arr), 2);
        assert_eq!(
            *get_element_from_matrix(
                row_index_2, col_index_2, num_rows, &arr),
            3);
    }*/
}
