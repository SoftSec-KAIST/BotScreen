#define NOMINMAX

#define ENCLAVE_FILE "C:\\dump\\SGX_MDL.signed.dll"
//#define ENCLAVE_FILE "C:\\Users\\okas832\\Desktop\\new_SGX\\Osiris-master\\Prerelease\\SGX_MDL.signed.dll"
#define DUMP_LOG 0
#define DUMP_PATH "C:\\dump\\"

#define SERVER_IP "192.168.0.6"
#define SERVER_PORT 55554

#include "SGX_MDL_u.h"
#include "sgx_urts.h"
#include "Dump.h"

#include "../SDK/GameEvent.h"
#include "../fnv.h"
#include "../Interfaces.h"
#include "../Memory.h"
#include "../SDK/Engine.h"
#include "../SDK/EngineTrace.h"
#include "../SDK/Entity.h"
#include "../SDK/EntityList.h"
#include "../SDK/LocalPlayer.h"
#include "../SDK/UserCmd.h"
#include "../SDK/Vector.h"
#include "../SDK/WeaponId.h"
#include "../SDK/GlobalVars.h"
#include "../SDK/PhysicsSurfaceProps.h"
#include "../SDK/WeaponData.h"
#include "../Config.h"
#include <windows.h>
#include <winbase.h>
#include <stdint.h>
#include <queue>
#include <map>
#include <string>
#include <math.h>
#include <concurrent_queue.h>

#pragma comment(lib,"Ws2_32.lib")

#include <winsock2.h>
#include <ws2tcpip.h>

#define WIN_SIZE 21
#define WIN_OFFSET 5

// initialized?
int init = 0;
// TODO
// When destory sockets and enclave?
// SGX global data
sgx_enclave_id_t eid;
sgx_status_t ret = SGX_SUCCESS;
sgx_launch_token_t token = { 0 };
int updated = 0;
int active = 1;

// socket
WSADATA wsaData = { 0 };
SOCKET hSocket;
SOCKADDR_IN servAddr;

struct sPlayer
{
    char* name;
    // XXX
    // Consider only there are two teams. Terror, Counter-T
    int team;
    double x, y, z;
    double sx, sy, sz;
};

struct Angle
{
    uint64_t time;
    double sx, sy, sz;
    double d_a;
    double d_p;
};
// window
std::map<std::string, std::queue<struct Angle>> wd;
// Process count left to each user
std::map<std::string, int> pcnt;

// Process Queue
Concurrency::concurrent_queue<std::pair<std::string, std::queue<struct Angle>>> pq;

DWORD dwThreadId;
HANDLE hThread;

static uint64_t GetSystemTime()
{
    const uint64_t UNIX_TIME_START = 0x019DB1DED53E8000;
    const uint64_t TICKS_PER_1MS = 10000;

    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);

    LARGE_INTEGER li;
    li.LowPart = ft.dwLowDateTime;
    li.HighPart = ft.dwHighDateTime;

    return (li.QuadPart - UNIX_TIME_START) / TICKS_PER_1MS;
}

static short getEntityIdByUserId(short UserId) noexcept
{
    for (int i = 1; i <= interfaces->engine->getMaxClients(); i++)
    {
        auto entity = interfaces->entityList->getEntity(i);
        if (PlayerInfo playerInfo; interfaces->engine->getPlayerInfo(i, playerInfo))
        {
            if (playerInfo.userId == UserId)
                return i;
        }
    }
    return -1;
}

void Dump::DumpEvent(GameEvent* event) noexcept
{
#if DUMP_LOG
    FILE* fout = fopen(DUMP_PATH "log_event.csv", "a+");
#endif
    PlayerInfo playerInfo1;
    switch (fnv::hashRuntime(event->getName()))
    {
        case fnv::hash("weapon_fire"):
            interfaces->engine->getPlayerInfo(getEntityIdByUserId(event->getInt("userid")), playerInfo1);
#if DUMP_LOG
            fprintf(fout, "%lld, %s\n", GetSystemTime(), playerInfo1.name);
#endif
            pcnt[playerInfo1.name] = 20;
            break;
    }
#if DUMP_LOG
    fclose(fout);
#endif
    return;
}

// Only considering N vs N game (N <= 12)
// XXX
// change value to interfaces->engine->getMaxClients()
#define MAXCLIENT 12

double sz_of_v(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

double l2norm3(double x, double y, double z)
{
    return sqrt(x * x + y * y + z * z);
}

void ClearQueue(std::queue<struct Angle>& q)
{
    std::queue<struct Angle> empty;
    std::swap(q, empty);
}

void QueueInput(std::string name, std::queue<struct Angle> q)
{
    if (pcnt[name] == 0)
        return;

    std::pair< std::string, std::queue<struct Angle> > t(name, q);
    pq.push(t);

    pcnt[name] -= 1;
}
DWORD WINAPI RunModel(LPVOID lpParam)
{
    while (1)
    {
        std::pair<std::string, std::queue<struct Angle>> now_ipt;
        if (!pq.try_pop(now_ipt))
            continue;

        double input[21 * 3] = { 0.0, };
        char output[0x400] = { 0, };
        size_t out_sz = 0;

        int i;
        for (i = 0; i < WIN_SIZE; i++)
        {
            auto front = now_ipt.second.front();
            input[i * 3 + 0] = front.sx;
            input[i * 3 + 1] = front.sy;
            input[i * 3 + 2] = front.sz;
            now_ipt.second.pop();
        }

        process(eid, (char *)now_ipt.first.c_str(), input, output, 0x400, &out_sz);

        send(hSocket, output, out_sz, 0);
    }
}


void Dump::DumpGameData() noexcept
{
    if (!init)
    {
        // init sgx enclave
        sgx_create_enclave(ENCLAVE_FILE, 1, &token, &updated, &eid, NULL);

        // init server connection
        WSAStartup(MAKEWORD(2, 2), &wsaData);
        hSocket = socket(PF_INET, SOCK_STREAM, 0);

        memset(&servAddr, 0, sizeof(servAddr));
        servAddr.sin_family = AF_INET;
        servAddr.sin_addr.s_addr = inet_addr(SERVER_IP);
        servAddr.sin_port = htons(SERVER_PORT);

        connect(hSocket, (SOCKADDR*)&servAddr, sizeof(servAddr));

        uint8_t enc[256] = { 0, };
        init_sgx(eid, enc);
        send(hSocket, (const char *)enc, 256, 0);

        hThread = CreateThread(NULL, 0, RunModel, NULL, 0, &dwThreadId);

        init = 1;
    }

    if (!localPlayer)
        return;

#if DUMP_LOG
    FILE* fout = fopen(DUMP_PATH "log_player.csv", "a+");
#endif

    uint64_t time = GetSystemTime();

    struct sPlayer player[MAXCLIENT];

    auto entity = localPlayer.get();
    const auto bonePosition = entity->getBonePosition(8);
    const auto eyeAngle = entity->eyeAngles();

    // collect raw data
    player[0].name = strdup(entity->getPlayerName().c_str());
    player[0].team = 1; //always team 1
    player[0].x = bonePosition.x;
    player[0].y = bonePosition.y;
    player[0].z = bonePosition.z;
    player[0].sx = eyeAngle.x;
    player[0].sy = eyeAngle.y;

#if DUMP_LOG
    fprintf(fout, "%lld, 0, %s, %f, %f, %f, %f, %f,", time, player[0].name, player[0].x, player[0].y, player[0].z, player[0].sx, player[0].sy);
#endif
    
    int pid = 1;
    int i, j;
    for (i = 1; i <= interfaces->engine->getMaxClients(); i++)
    {
        auto entity = interfaces->entityList->getEntity(i);
        if (!entity || entity == localPlayer.get() || entity->isDormant() || !entity->isAlive())
            continue;
        PlayerInfo playerInfo;
        interfaces->engine->getPlayerInfo(entity->index(), playerInfo);
        const auto bonePosition = entity->getBonePosition(8);
        const auto eyeAngle = entity->eyeAngles();

        if (((unsigned char*)entity->getPlayerName().c_str())[0] == 0xEB) // bot filter
            player[pid].name = strdup(entity->getPlayerName().c_str() + 4);
        else
            player[pid].name = strdup(entity->getPlayerName().c_str());
        player[pid].team = localPlayer->isOtherEnemy(entity); // if same team : 1, if not : 0  lazy trick
        player[pid].x = bonePosition.x;
        player[pid].y = bonePosition.y;
        player[pid].z = bonePosition.z;
        player[pid].sx = eyeAngle.x;
        player[pid].sy = eyeAngle.y;
#if DUMP_LOG
        fprintf(fout, "1, %s, %f, %f, %f, %f, %f,", player[pid].name, player[pid].x, player[pid].y, player[pid].z, player[pid].sx, player[pid].sy);
#endif
        pid++;
    }
#if DUMP_LOG
    fprintf(fout, "\n");
    fclose(fout);
#endif

    for (i = 0; i < pid; i++)
    {
        if (player[i].sx > 90.0) player[i].sx -= 360.0;
        else if (player[i].sx < -90.0) player[i].sx += 360.0;
        player[i].sx = 90.0 - player[i].sx;

        if (player[i].sy < 0.0) player[i].sy += 360.0;
        else if (player[i].sy >= 360.0) player[i].sy -= 360.0;
        
        player[i].sx *= M_PI;
        player[i].sx /= 180.0;
        player[i].sy *= M_PI;
        player[i].sy /= 180.0;

        double tx, ty, tz;

        tx = cos(player[i].sy) * sin(player[i].sx);
        ty = sin(player[i].sy) * sin(player[i].sx);
        tz = cos(player[i].sx);

        player[i].sx = tx;
        player[i].sy = ty;
        player[i].sz = tz;
    }
    
    for (i = 0; i < pid; i++) // obs
    {
        double d_a = INFINITY;
        double d_p = INFINITY;
        for (j = 0; j < pid; j++) // tar
        {
            if (i == j || player[i].team == player[j].team) continue;

            double dt_x, dt_y, dt_z;
            dt_x = player[j].x - player[i].x;
            dt_y = player[j].y - player[i].y;
            dt_z = player[j].z - player[i].z;

            // cosine distance
            double dis = dt_x * player[i].sx + dt_y * player[i].sy + dt_z * player[i].sz;
            dis /= sz_of_v(dt_x, dt_y, dt_z);
            dis /= sz_of_v(player[i].sx, player[i].sy, player[i].sz);
            dis = 1.0 - dis;
            if (d_a > dis && dis < 0.15)
            {
                d_a = dis;
                d_p = l2norm3(dt_y * player[i].sz - dt_z * player[i].sy,
                              dt_z * player[i].sx - dt_x * player[i].sz,
                              dt_x * player[i].sy - dt_y * player[i].sx)
                      / l2norm3(player[i].sx, player[i].sy, player[i].sz);
            }
        }
        
        //if (d_a != INFINITY && d_p != INFINITY)
        {
            auto it = wd.insert({ player[i].name, std::queue<struct Angle>() });

            struct Angle tmp = {
                .time = time,
                .sx = player[i].sx,
                .sy = player[i].sy,
                .sz = player[i].sz,
                .d_a = d_a,
                .d_p = d_p
            };

            // base delta 16ms
            // lower than 16ms - ignore
            // 16ms - push
            // 17ms~79ms - interpolation
            // higher than 80 - clean queue and push

            if (it.first->second.empty())
            {
                it.first->second.push(tmp);
            }
            else
            {
                auto last_frame = it.first->second.back();
                auto time_delta = time - last_frame.time;
                if (time_delta == 16)
                {
                    it.first->second.push(tmp);
                    while (it.first->second.size() > WIN_SIZE + WIN_OFFSET)
                    {
                        it.first->second.pop();
                    }
                    if (it.first->second.size() == WIN_SIZE + WIN_OFFSET)
                    {
                        QueueInput(player[i].name, it.first->second);
                    }
                }
                else if (time_delta > 16 && time_delta < 80)
                {
                    double delta_sx = (tmp.sx - last_frame.sx) / time_delta;
                    double delta_sy = (tmp.sy - last_frame.sy) / time_delta;
                    double delta_sz = (tmp.sz - last_frame.sz) / time_delta;
                    double delta_da = (tmp.d_a - last_frame.d_a) / time_delta;
                    double delta_dp = (tmp.d_p - last_frame.d_p) / time_delta;

                    for (j = 1; j <= time_delta / 16; j++)
                    {
                        tmp = {
                            .time = last_frame.time + j * 16,
                            .sx = last_frame.sx + delta_sx * j,
                            .sy = last_frame.sy + delta_sy * j,
                            .sz = last_frame.sz + delta_sz * j,
                            .d_a = last_frame.d_a + delta_da * j,
                            .d_p = last_frame.d_p + delta_dp * j
                        };
                        it.first->second.push(tmp);
                        while (it.first->second.size() > WIN_SIZE + WIN_OFFSET)
                        {
                            it.first->second.pop();
                        }
                        if (it.first->second.size() == WIN_SIZE + WIN_OFFSET)
                        {
                            QueueInput(player[i].name, it.first->second);
                        }
                    }
                }
                else if (time_delta >= 80)
                {
                    ClearQueue(it.first->second);
                    it.first->second.push(tmp);
                }
            }
        }
    }
}