#include "SGX_MDL_u.h"
#include <errno.h>

typedef struct ms_process_t {
	char* ms_username;
	double* ms_input;
	char* ms_output_buf;
	size_t ms_len;
	size_t* ms_res_size;
} ms_process_t;

typedef struct ms_init_sgx_t {
	uint8_t* ms_enc;
} ms_init_sgx_t;

typedef struct ms_sgx_oc_cpuidex_t {
	int* ms_cpuinfo;
	int ms_leaf;
	int ms_subleaf;
} ms_sgx_oc_cpuidex_t;

typedef struct ms_sgx_thread_wait_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_self;
} ms_sgx_thread_wait_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_set_untrusted_event_ocall_t {
	int ms_retval;
	const void* ms_waiter;
} ms_sgx_thread_set_untrusted_event_ocall_t;

typedef struct ms_sgx_thread_setwait_untrusted_events_ocall_t {
	int ms_retval;
	const void* ms_waiter;
	const void* ms_self;
} ms_sgx_thread_setwait_untrusted_events_ocall_t;

typedef struct ms_sgx_thread_set_multiple_untrusted_events_ocall_t {
	int ms_retval;
	const void** ms_waiters;
	size_t ms_total;
} ms_sgx_thread_set_multiple_untrusted_events_ocall_t;

static sgx_status_t SGX_CDECL SGX_MDL_sgx_oc_cpuidex(void* pms)
{
	ms_sgx_oc_cpuidex_t* ms = SGX_CAST(ms_sgx_oc_cpuidex_t*, pms);
	sgx_oc_cpuidex(ms->ms_cpuinfo, ms->ms_leaf, ms->ms_subleaf);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL SGX_MDL_sgx_thread_wait_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_wait_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_wait_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_wait_untrusted_event_ocall(ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL SGX_MDL_sgx_thread_set_untrusted_event_ocall(void* pms)
{
	ms_sgx_thread_set_untrusted_event_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_untrusted_event_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_untrusted_event_ocall(ms->ms_waiter);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL SGX_MDL_sgx_thread_setwait_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_setwait_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_setwait_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_setwait_untrusted_events_ocall(ms->ms_waiter, ms->ms_self);

	return SGX_SUCCESS;
}

static sgx_status_t SGX_CDECL SGX_MDL_sgx_thread_set_multiple_untrusted_events_ocall(void* pms)
{
	ms_sgx_thread_set_multiple_untrusted_events_ocall_t* ms = SGX_CAST(ms_sgx_thread_set_multiple_untrusted_events_ocall_t*, pms);
	ms->ms_retval = sgx_thread_set_multiple_untrusted_events_ocall(ms->ms_waiters, ms->ms_total);

	return SGX_SUCCESS;
}

static const struct {
	size_t nr_ocall;
	void * func_addr[5];
} ocall_table_SGX_MDL = {
	5,
	{
		(void*)(uintptr_t)SGX_MDL_sgx_oc_cpuidex,
		(void*)(uintptr_t)SGX_MDL_sgx_thread_wait_untrusted_event_ocall,
		(void*)(uintptr_t)SGX_MDL_sgx_thread_set_untrusted_event_ocall,
		(void*)(uintptr_t)SGX_MDL_sgx_thread_setwait_untrusted_events_ocall,
		(void*)(uintptr_t)SGX_MDL_sgx_thread_set_multiple_untrusted_events_ocall,
	}
};

sgx_status_t process(sgx_enclave_id_t eid, char* username, double* input, char* output_buf, size_t len, size_t* res_size)
{
	sgx_status_t status;
	ms_process_t ms;
	ms.ms_username = username;
	ms.ms_input = input;
	ms.ms_output_buf = output_buf;
	ms.ms_len = len;
	ms.ms_res_size = res_size;
	status = sgx_ecall(eid, 0, &ocall_table_SGX_MDL, &ms);
	return status;
}

sgx_status_t init_sgx(sgx_enclave_id_t eid, uint8_t* enc)
{
	sgx_status_t status;
	ms_init_sgx_t ms;
	ms.ms_enc = enc;
	status = sgx_ecall(eid, 1, &ocall_table_SGX_MDL, &ms);
	return status;
}

