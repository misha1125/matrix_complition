#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <lapacke.h>

using namespace std;

extern "C"{
void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*);
void dgemm_(char*, char*, int*,int*,int*, double*, double*, int*,double*,int*,double*,double*,int*);
}

class SparceMatrix{
public:
    void mul_to_transpose_matrix(const double *matrix, double * res, int second_size) const{
        for (int i = 0; i < size; ++i) {
            for (int k = 0; k < second_size; ++k) {
                //res[i,k] = 0
                res[k * size + i] = 0;
                for (int j = row_start[i]; j < row_start[i+1]; ++j) {
                    //res[i,k] += Sp[i,j] * matrixT[k, j]
                    res[k * size + i] += vals_row[j] *
                                         matrix[indexes_in_row[j] * second_size + k];
                }
            }
        }
    }

    void mul_transpose_to_matrix(const double *matrix,double * res, int first_size) const{
        for (int k = 0; k < first_size; ++k) {
            for (int i = 0; i < size; ++i) {
                //res[k,i] = 0
                res[i * first_size + k] = 0;
                for (int j = column_start[i]; j < column_start[i+1]; ++j) {
                    //res[k,i] += Sp[i,j]*matrixT[j,k]
                    res[i * first_size + k] += vals_col[j] *
                                         matrix[indexes_in_column[j]  + k * size];
                }
            }
        }
    }
    double *vals_row = nullptr;
    double *vals_col = nullptr;
    static int* indexes_in_row;
    static int* row_start;
    static int* indexes_in_column;
    static int* column_start;
    static int size;
    static void create_mask(double p, int sz){
        vector<int> indexes_row;
        SparceMatrix::size = sz;
        row_start = new int[sz + 1];
        column_start = new int[sz + 1];
        row_start[0] = 0;
        column_start[0] = 0;
        for (int i = 0; i < sz; ++i) {
            for (int j = 0; j < sz; ++j) {
                if ((double)rand()/RAND_MAX < p) {
                    indexes_row.push_back(j);
                }
            }
            row_start[i + 1] = indexes_row.size();
        }
        indexes_in_row = new int [indexes_row.size()];
        for (int i = 0; i < indexes_row.size(); ++i) {
            indexes_in_row[i] = indexes_row[i];
        }
        indexes_in_column = new int[indexes_row.size()];
        int last_index = 0;
        for (int find_index = 0; find_index < size; ++find_index) {
            for (int i = 0; i < size; ++i){
                for (int j = row_start[i];j<row_start[i+1]; ++j){
                    if (indexes_in_row[j] == find_index) {
                        indexes_in_column[last_index++] = i;
                    }
                }
            }
            column_start[find_index+1] = last_index;
        }
    }
    void fill_matrix(const double *U, const double *V, int rank){
        if (vals_row == nullptr) {
            vals_row = new double [row_start[size]];
            vals_col = new double [column_start[size]];
        }
        for (int i = 0; i < size; ++i) {
            for (int j = row_start[i]; j < row_start[i+1]; ++j) {
                vals_row[j] = 0;
                for (int k = 0; k < rank; ++k) {
                    //Sp[i,j] += U[i,k] * V[k, j]
                    vals_row[j] += U[k * size + i] * V[indexes_in_row[j] * rank + k];
                }
            }
        }
        for (int j = 0; j < size; ++j) {
            for (int i = column_start[j]; i < column_start[j+1]; ++i) {
                vals_col[i] = 0;
                for (int k = 0; k < rank; ++k) {
                    //Sp[i,j] += U[i,k] * V[k,j]
                    vals_col[i] += U[k * size + indexes_in_column[i]] * V[j * rank + k];
                }
            }
        }
    }
    void minus_matrix(const SparceMatrix &other){
        for (int i = 0; i < row_start[size];++i) {
            vals_row[i] -= other.vals_row[i];
            vals_col[i] -= other.vals_col[i];
        }
    }
    void mul_matrix(double other) {
        for (int i = 0; i < row_start[size];++i) {
            vals_row[i] *= other;
            vals_col[i] *= other;
        }
    }
    ~SparceMatrix(){
        delete[]vals_row;
    }
    double fro_norm() const{
        double sum = 0;
        for (int i = 0 ;i<row_start[size];++i){
            sum += vals_row[i] * vals_row[i];
        }
        return sqrt(sum);
    }
} step_matrix, source_matrix;
int *SparceMatrix::indexes_in_row, *SparceMatrix::indexes_in_column;
int *SparceMatrix::row_start, *SparceMatrix::column_start;
int SparceMatrix::size;

double * allocate_matrix(int size_1, int size_2) {
    return new double[size_1*size_2];
}

void deallocate_matrix(double* matrix) {
    delete [] matrix;
}

void print_sparce(const SparceMatrix& matrix) {
    for (int i = 0; i < SparceMatrix::size; ++i) {
        std::cout<<i<<":     ";
        for (int j = SparceMatrix::row_start[i]; j < SparceMatrix::row_start[i+1]; ++j){
            std::cout << setw(6) << setprecision(2) << SparceMatrix::indexes_in_row[j] << ":" << matrix.vals_row[j] << " ";
        }
        std::cout<<std::endl;
    }
}

void print_matrix(double * matrix, int m, int n){
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j){
            std::cout<<setw(6)<<setprecision(2)<<matrix[j*m + i]<<" ";
        }
        std::cout<<std::endl;
    }
}

double * mult (const double * first,const double * second, int size_one,
                             int common_size, int size_two) {
    double * res = allocate_matrix(size_one,size_two);
    for (int j = 0; j < size_two; ++j) {
        for (int i = 0; i < size_one; ++i) {
            res[i + j*size_one] = 0;
            for (int k = 0; k < common_size; ++k) {
                res[i + j*size_one] += first[i + k*size_one]*second[k + j*common_size];
            }
        }
    }
    return res;
}

double *generate_rand(size_t n, size_t m) {
    double * res = allocate_matrix(n, m);
    for (size_t i = 0; i < m; ++i){
        for (size_t j = 0; j < n; ++j){
            res[i*n + j] = static_cast<double>(rand() % 200 - 100) / 100;
        }
    }
    return res;
}

void minus_matrix(double *first, const double *second, int len) {
    for (int i = 0; i< len;++i){
        first[i] -= second[i];
    }
}

double *copy(double * matrix, int len){
    auto copy = new double [len];
    for (int i = 0; i < len; ++i){
        copy[i] = matrix[i];
    }
    return copy;
}

double* Transpose(double * matrix, int m, int n) {
    auto res = new double[m*n];
    for(int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            //res[j,i] =  ar[i,j]
            res[i * n + j] = matrix[j * m + i];
        }
    }
    return res;
}

pair<double*, double*> QR(double *matrix, int m, int n) {
    double * tau = new double [n];
    int lwork = 64*m;
    double * work = new double [lwork];
    int info;
    //double *A = copy(matrix, m*n);

//  void dgeqrf_(int*, int*, double*, int*, double*, double*, int*, int*)
    dgeqrf_(&m, &n, matrix, &m, tau, work, &lwork, &info);
    double *r = new double [n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0 ; j < n; ++j) {
            if (j < i){
                r[j * n + i] = 0;
            } else {
                r[j * n + i] = matrix[j * m + i];
            }
        }
    }
    /*print_matrix(matrix, m, n);
    std::cout<<std::endl;
    print_matrix(r, n, n);*/
    dorgqr_(&m, &n, &n, matrix, &m, tau, work, &lwork, &info);
    return {matrix, r};
}

int main()
{
    //srand(1651);
    int n = 1024;
    int rank = 10;
    double eps = 0.00001;
    double  p = 0.2;
    double step_size = (3.0/4.0)/p;
    SparceMatrix::create_mask(p, n);
    double * U = generate_rand(n, rank);
    double * V_T = generate_rand(rank, n);
    source_matrix.fill_matrix(U, V_T, rank);
    int step_cnt = 50;
    deallocate_matrix(U);
    deallocate_matrix(V_T);
    U = generate_rand(n, rank);
    V_T = generate_rand(rank, n);
    step_matrix.fill_matrix(U,V_T,rank);
    step_matrix.minus_matrix(source_matrix);
    std::cout<<step_matrix.fro_norm()<<"\n";
    auto tmp = allocate_matrix(n,rank);
    while (step_cnt--) {
        //шаг для U
        auto res = QR(U,n,rank);
        U = res.first;
        auto V_was = V_T;
        //V.T = R*V.T
        V_T = mult(res.second, V_T, rank, rank, n);
        delete[]res.second;
        delete[]V_was;

        //U.T@tau(U@V.T - T)
        step_matrix.fill_matrix(U,V_T,rank);
        step_matrix.minus_matrix(source_matrix);
        step_matrix.mul_matrix(step_size);
        step_matrix.mul_transpose_to_matrix(U, tmp, rank);
        minus_matrix(V_T, tmp, n * rank);

        //шаг для V
        auto V = Transpose(V_T,rank,n);
        delete[]V_T;
        res = QR(V,n,rank);
        V_T = Transpose(res.first, n, rank);
        delete [] res.first;
        auto U_old = U;
        auto R_ = Transpose(res.second,rank,rank);
        delete[]res.second;
        U = mult(U,R_,n,rank,rank);
        delete[]U_old;

        step_matrix.fill_matrix(U,V_T,rank);
        step_matrix.minus_matrix(source_matrix);
        step_matrix.mul_matrix(step_size);
        step_matrix.mul_to_transpose_matrix(V_T, tmp, rank);
        minus_matrix(U, tmp, n * rank);

        step_matrix.fill_matrix(U,V_T,rank);
        step_matrix.minus_matrix(source_matrix);
        double norm = step_matrix.fro_norm();
        std::cout<<norm<<"\n";
        if (norm < eps) {
            break;
        }
    }
    deallocate_matrix(U);
    deallocate_matrix(V_T);
    deallocate_matrix(tmp);
    delete []SparceMatrix::row_start;
    delete []SparceMatrix::indexes_in_row;
    delete []SparceMatrix::column_start;
    delete []SparceMatrix::indexes_in_column;
    return 0;
}