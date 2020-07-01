#include "mpi.h"
#include "stdio.h"
#include "unistd.h"

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	printf("rank is %d, size is %d\n", rank, size);
	sleep(1);

	int msg;
	if (!rank)
	{
		msg = 1337;
		for (int id = 1; id < size; id++) 
			MPI_Send(&msg, 1, MPI_INT, id, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Status stat;
		MPI_Recv(&msg, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
		printf("rank %d got message: %d\n", rank, msg);
	}

	MPI_Finalize();
	return 0;
}
