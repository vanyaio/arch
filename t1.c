#include "mpi.h"
#include "stdio.h"
#include "unistd.h"

int main(int argc, char **argv)
{
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (!rank)
		printf("info\n");
	sleep(0.5);
	printf("rank is %d, size is %d\n", rank, size);

	sleep(0.5);

	int msg = 1337;
	int src = (rank + 1) % size;
	MPI_Status stat;
	if (!rank)
	{
		printf("circle\n");
		MPI_Send(&msg, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
		printf("rank %d send message to rank %d\n", rank, src);
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
		printf("rank %d recv message from rank %d\n", rank, size - 1);
	}
	else
	{
		sleep(0.5);
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
		printf("rank %d recv message from rank %d\n", rank, rank - 1);
		MPI_Send(&msg, 1, MPI_INT, src, 0, MPI_COMM_WORLD);
		printf("rank %d send message to rank %d\n", rank, src);
	}


	sleep(0.5);
	if (!rank)
	{
		printf("master->slave\n");
		for (int id = 1; id < size; id++)
			MPI_Send(&msg, 1, MPI_INT, id, 0, MPI_COMM_WORLD);
	}
	else
	{
		MPI_Recv(&msg, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
		printf("rank %d got message: %d\n", rank, msg);
	}


	sleep(3);
	if (!rank)
		printf("\n\n\neach->each\n");

	for (int id = 0; id < size; id++)
	{
		if (id != rank)
			MPI_Send(&msg, 1, MPI_INT, id, 0, MPI_COMM_WORLD);
	}
	for (int i = 0; i < size - 1; i++)
	{
		MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
		printf("rank %d got message: %d\n", rank, msg);
	}

	MPI_Finalize();
	return 0;
}
