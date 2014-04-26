#include<iostream>
#include<sqlite3.h>
#include<cstdlib>
#include<string>
#include<cstring>
#include<fstream>
#include<sstream>
#include<sys/stat.h>
#include<cassert>
 
using namespace std;
 
const int KeyWordLen=60;    //“概念”的最大长度
const int POSLen=8;     //“词性”的最大长度
const int SememeSetLen=200;     //一个“概念”对应的“义原”集合的最大长度
 
int main(int argc,char *argv[]){
    sqlite3 *db;
    char *zErrMsg=0;
    int rc;
    rc=sqlite3_open("glossary.db",&db);     //打开数据库
    assert(rc==SQLITE_OK);
    char sql[500]={0};
    sprintf(sql,"create table t_gloss(id integerprimary key,concept varchar(%d),pos char(%d),semset varchar(%d))",KeyWordLen,POSLen,SememeSetLen);
    rc=sqlite3_exec(db,sql,0,0,&zErrMsg);       //创建表
    assert(rc==SQLITE_OK);
     
    ifstream ifs("glossary.dat");   //打开词典文件
    assert(ifs);
    string line;
    int recordid=0;
    while(getline(ifs,line)){       //逐行读取词典文件
        istringstream stream(line);
        string word,pos,sememe;
        stream>>word>>pos>>sememe;  //由空白把一行分割成：词、词性、义原集合
        string set;
        if(sememe[0]=='{'){ //该行是虚词，因为虚词的描述只有“{句法义原}”或“{关系义原}”
            set=sememe+",";
        }
        else{       //该行是实词，要按“基本义原描述式\n其他义原描述式\n关系义原描述式\n关系符号义原描述式”存储
            string str1,str2,str3,str4;
            string::size_type pos1,pos2;
            pos1=0;
            bool flag=true;
            while(flag){
                pos2=sememe.find(",",pos1);
                string sem;
                if(string::npos==pos2){     //已是最后一个义原
                    flag=false;
                    sem=sememe.substr(pos1);    //提取最后一个义原
                }
                else{
                    sem=sememe.substr(pos1,pos2-pos1);  //提取下一个义原
                }
                pos1=pos2+1;
                 
                if(sem.find("=")!=string::npos){        //关系义原,加入str3
                    str3+=sem+",";
                }
                else{
                    char c=sem[0];
                    if((c>64&&c<91) || (c>96&&c<123) || (c==40)){       //义原以大/小写英文字母开始，或者是具体词--单独在小括号里，属于其他义原，加入str2。40是"("的ASCII值
                        str2+=sem+",";
                    }
                    else{       //关系符号义原，加入str4
                        str4+=sem+",";
                    }
                }
            }
            //把str2中的第一条取出来，赋给str1
            string::size_type pos3=str2.find(",");
            if(pos3!=string::npos){
                str1=str2.substr(0,pos3+1);
                str2.erase(0,pos3+1);
            }
            set=str1+"\n"+str2+"\n"+str3+"\n"+str4;
        }
        bzero(sql,sizeof(sql));
        sprintf(sql,"insert into t_gloss values(%d,\'%s\',\'%s\',\'%s\')",recordid++,word.c_str(),pos.c_str(),set.c_str());
        rc=sqlite3_exec(db,sql,0,0,&zErrMsg);
        assert(rc==SQLITE_OK);
    }
    ifs.close();
    //在“概念”上建立索引。以后要经常依据“概念”进行查询
    bzero(sql,sizeof(sql));
    sprintf(sql,"create index index1 on t_gloss(concept)");
    rc=sqlite3_exec(db,sql,0,0,&zErrMsg);
    assert(rc==SQLITE_OK);
     
    sqlite3_close(db);
    return 0;
}
