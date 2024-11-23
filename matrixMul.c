#include <stdio.h>
#include <stdlib.h> 
#include <mpi.h>  

#define N 1000 

void displayMatrix(int** matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void matrixMultiply(int** A, int** B, int** C, int n, int startRow, int endRow) {
    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int ** allocateMatrix( int n ){
    int** matrix = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; ++i){
        matrix[i] =  (int*)malloc(n * sizeof(int));
    }
    return matrix;
}

int main(int argc, char** argv) {
    int rank, size;
    int rowsPerProc = N / size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   
    // Dynamically allocate memory for the matrices
    int** A = allocateMatrix(N);
    int** B = allocateMatrix(N);
    int** C = allocateMatrix(N);
    int** localA = allocateMatrix(rowsPerProc);
    int** localC = allocateMatrix(rowsPerProc);
    
   

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    printf("Matrices allocated successfully.\n");
    
    
    // Initialize matrices A and B

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
            A[i][j] = 1;
            B[i][j] = 1;
            C[i][j] = 0;
            localA[i][j] = 0;
            localC[i][j] = 0;
            }
        }

    
       
    
    printf("Matrices initialized successfully.\n");

    double start_time = MPI_Wtime();

    // Scatter rows of A 
    for (int i =0; i < rowsPerProc; ++i){
        MPI_Scatter(A[rank * rowsPerProc + i], N, MPI_INT, localA[i], N, MPI_INT, 0, MPI_COMM_WORLD);
    }


    // Broadcast B to all processes
    for (int i =0; i < N; ++i){
    MPI_Bcast(&(B[i]), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    }
   
    // Run matrixMultiply with the local rows of A
    matrixMultiply(localA, B, localC, N, rank * rowsPerProc, (rank + 1) * rowsPerProc);
    
    // Gather all rows of C
    for (int i = 0; i < rowsPerProc; ++i){
        MPI_Gather(localC[i], N, MPI_INT, C[rank * rowsPerProc + i], N, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    printf("Matrix multiplication complete!\n");
    double end_time = MPI_Wtime();
    double elapased_time = end_time - start_time;
    printf ("Elapsed time: %f seconds\n", elapased_time);

    //Optionally display the resulting matrix C
    // printf("Resulting Matrix C:\n");
    // displayMatrix(C, N);

    // Free dynamically allocated memory
    for (int i = 0; i < N; ++i) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
        free(localA[i]);
    }
    free(A);
    free(B);
    free(C);
    free(localA);

    MPI_Finalize();
    return 0;
}
